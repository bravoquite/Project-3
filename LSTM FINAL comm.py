from os import system
system('cls')
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.animation as animation


torch.manual_seed(15)

#starting on cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

results = []  #results for CSV

ds = xr.open_dataset(r"C:\\Data\\gistemp250_GHCNv4.nc")

zones = [
    (0, 22.5),
    (22.5, 45),
    (45, 67.5),
    (67.5, 90)
]

#model definition
def create_lstm_model(input_size=3, hidden_size=64, num_layers=2):
    class LSTMTempModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTMTempModel().to(device)
#sequences generator
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

sequence_length = 36 #good sequence for numerous data in long time period 
future_years = 2050 - 2024
future_steps = future_years * 12

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
#training and testing
for idx, (lat_min, lat_max) in enumerate(zones):
    train_zone = ds['tempanomaly'].sel(lat=slice(lat_min, lat_max))
    test_zone = ds['tempanomaly'].sel(lat=slice(-lat_max, -lat_min))

    pts = []
    for lat in train_zone.lat.values:
        for lon in train_zone.lon.values:
            series = train_zone.sel(lat=lat, lon=lon).values
            if not np.isnan(series).all():
                pts.append((np.count_nonzero(~np.isnan(series)), lat, lon))

    pts.sort(reverse=True)

    
    #we are taking 10% of available point for training
    sample_fraction = 0.1  
    sample_size = max(1, int(len(pts) * sample_fraction))
    sampled_pts = [(lat, lon) for _, lat, lon in pts[:sample_size]]

    trains = [train_zone.sel(lat=lat, lon=lon) for lat, lon in sampled_pts]
    train_series = sum(trains) / len(trains) 
    
    #we are looking for best city for testing (by number of data)
    pts_test = []
    for lat in test_zone.lat.values:
        for lon in test_zone.lon.values:
            series = test_zone.sel(lat=lat, lon=lon).values
            if not np.isnan(series).all():
                pts_test.append((np.count_nonzero(~np.isnan(series)), lat, lon))
    if not pts_test:
        print(f"No valid test point in zone {lat_min}-{lat_max}S, skipping.")
        continue
    lat_t, lon_t = max(pts_test)[1:]
    test_series = test_zone.sel(lat=lat_t, lon=lon_t)

    time = ds['time'].values
    years = np.array([t.astype('datetime64[M]').astype(int) // 12 for t in time]).reshape(-1, 1).astype(np.float32)
    months = np.array([t.astype('datetime64[M]').astype(int) % 12 + 1 for t in time]).reshape(-1, 1).astype(np.float32)

    y_train = train_series.values.astype(np.float32).reshape(-1, 1)
    y_test = test_series.values.astype(np.float32).reshape(-1, 1)

    #NaN filter
    mask = (~np.isnan(y_train)) & (~np.isnan(y_test))
    mask_flat = mask.flatten()
    yt = y_train[mask_flat].reshape(-1, 1)
    yq = y_test[mask_flat].reshape(-1, 1)
    yr = years[mask_flat]
    mo = months[mask_flat]

    #protection for not enought data
    if len(yt) <= sequence_length:
        print(f"Not enough data in zone {lat_min}-{lat_max} to train model. Skipping.")
        continue
    
    #scalling (normalization)
    sy = StandardScaler().fit(yt)
    yt_norm = sy.transform(yt)
    yr_scaler = StandardScaler().fit(yr)
    mo_scaler = StandardScaler().fit(mo)
    yr_norm = yr_scaler.transform(yr)
    mo_norm = mo_scaler.transform(mo)

    Xs_combined = np.hstack((yt_norm, yr_norm, mo_norm))

    X_seq, y_seq = create_sequences(Xs_combined, yt_norm, sequence_length)
    Xt = torch.tensor(X_seq, dtype=torch.float32).to(device)
    yt_t = torch.tensor(y_seq, dtype=torch.float32).to(device)

    #launching model and training fo 700 epochs
    model = create_lstm_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    for ep in range(700):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt_t)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Zone {lat_min}-{lat_max} - Epoch {ep}, Loss: {loss.item():.6f}")
    
    #predicting
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(Xs_combined[-sequence_length:].reshape(1, sequence_length, 3), dtype=torch.float32).to(device)
        future_preds_scaled = []
        yr_base = yr[-1][0]
        mo_base = mo[-1][0]

        for i in range(1, future_steps + 1):
            future_year = yr_base + (mo_base + i - 1) // 12
            future_month = (mo_base + i - 1) % 12 + 1
            pred_val = future_preds_scaled[-1] if future_preds_scaled else last_seq[0, -1, 0].item()
            t_feat = np.array([[pred_val, future_year, future_month]], dtype=np.float32)
            t_feat[:, 1] = yr_scaler.transform(t_feat[:, 1].reshape(-1, 1)).flatten()
            t_feat[:, 2] = mo_scaler.transform(t_feat[:, 2].reshape(-1, 1)).flatten()
            t_feat_tensor = torch.tensor(t_feat.reshape(1, 1, 3), dtype=torch.float32).to(device)
            last_seq = torch.cat((last_seq[:, 1:, :], t_feat_tensor), dim=1)
            next_val = model(last_seq)
            future_preds_scaled.append(next_val.cpu().numpy()[0, 0])

        future_preds = sy.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

        Xf_seq = np.array([Xs_combined[i:i + sequence_length] for i in range(len(Xs_combined) - sequence_length)])
        Xf_t = torch.tensor(Xf_seq, dtype=torch.float32).to(device)
        past_preds_scaled = model(Xf_t).cpu().numpy()
        past_preds = sy.inverse_transform(past_preds_scaled).flatten()

    #fitting and prediction
    start_year = 1880 + sequence_length / 12
    pred_start_year = 2024 + 1 / 12
    pred_years = pred_start_year + np.arange(future_steps) / 12
    past_years = np.arange(start_year, start_year + len(past_preds) / 12, 1 / 12)

    
    zone_label = f"{lat_min}-{lat_max}N"
    df = pd.DataFrame({
        'zone': zone_label,
        'year': list(past_years) + list(pred_years),
        'type': ['fit'] * len(past_years) + ['forecast'] * len(pred_years),
        'temperature': list(past_preds) + list(future_preds)
    })
    results.append(df)
    #plots
    axs[idx].plot(past_years, yq[sequence_length:].flatten(), label=f'Test city ({lat_t:.1f},{lon_t:.1f})', alpha=0.9)
    axs[idx].plot(past_years, past_preds, label='Model fit', linestyle='--', linewidth=2.0)
    axs[idx].plot(pred_years, future_preds, label='LSTM forecast to 2050', linewidth=2.2)
    axs[idx].set_title(f"Zone {lat_min}–{lat_max}N MSE={mean_squared_error(yq[sequence_length:].flatten(), past_preds):.3f}")
    axs[idx].legend()
    axs[idx].grid(True)
#saving fitting and predition into CSV file
pd.concat(results).to_csv("temperature_predictions_1880_2050.csv", index=False)

plt.tight_layout()
plt.show()

#second part: comparison

#loading data from CSV
df = pd.read_csv(r"C:\Users\wdk33\temperature_predictions_1880_2050.csv")

#preparing data
zones = df['zone'].unique()
zones.sort()

#MLP model
class TempModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

#plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

#animation lists
lines_nn = []
lines_poly = []
lines_exp = []
x_full_list = []
y_mlp_full_list = []
y_poly_full_list = []
y_exp_full_list = []
y_real_list = []

#loop through zones
for i, zone in enumerate(zones):
    df_zone = df[df['zone'] == zone]
    
    df_zone_full = df_zone[df_zone['type'].isin(['fit', 'forecast'])].copy()

    x_years = df_zone_full['year'].values.reshape(-1, 1).astype(np.float32)
    y_values = df_zone_full['temperature'].values.reshape(-1, 1).astype(np.float32)

    #scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(x_years)
    y_scaled = scaler_y.fit_transform(y_values)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    #launching MLP
    model = TempModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    #prediction to 2050
    x_full = np.arange(df['year'].min(), 2050 + 1/12, 1/12).astype(np.float32).reshape(-1, 1)

    X_full_scaled = scaler_x.transform(x_full)
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred_scaled_full = model(X_full_tensor).numpy()
        y_pred_mlp_full = scaler_y.inverse_transform(y_pred_scaled_full).flatten()

    #polynomial fit
    x_flat = x_years.flatten()
    y_flat = y_values.flatten()

    best_r2 = -np.inf
    best_poly = None
    best_degree = 0

    for degree in range(1, 21):
        polyfit = np.poly1d(np.polyfit(x_flat, y_flat, degree))
        y_poly = polyfit(x_full.flatten())
        r2 = r2_score(y_flat, polyfit(x_flat))
        
        if r2 > best_r2:
            best_r2 = r2
            best_poly = polyfit
            best_degree = degree
            y_poly_best_full = y_poly

    #exponencial fit
    offset = abs(np.min(y_flat)) + 1.0
    log_y = np.log(y_flat + offset)

    coeffs = np.polyfit(x_flat, log_y, 1)
    b = coeffs[0]
    ln_a = coeffs[1]
    a = np.exp(ln_a)

    y_exp_full = a * np.exp(b * x_full.flatten()) - offset

    #plot and animation
    ax = axes[i]

    #real data is static in the background
    ax.plot(x_flat, y_flat, label='Real data (fit + forecast)', color='lightgray', alpha=1.0)

    #animated lines initialization
    line_nn, = ax.plot([], [], label='Neural Network (MLP)', color='blue')
    line_poly, = ax.plot([], [], label=f'Polynomial (deg. {best_degree})', color='red', linestyle=':')
    line_exp, = ax.plot([], [], label='Exponential', color='green', linestyle='-.')

    #animation references
    lines_nn.append(line_nn)
    lines_poly.append(line_poly)
    lines_exp.append(line_exp)
    x_full_list.append(x_full.flatten())
    y_mlp_full_list.append(y_pred_mlp_full)
    y_poly_full_list.append(y_poly_best_full)
    y_exp_full_list.append(y_exp_full)
    y_real_list.append(y_flat)

    #axes 
    ax.set_xlim(1880, 2050)
    ax.set_ylim(min(y_flat) - 0.5, max(max(y_pred_mlp_full), max(y_poly_best_full), max(y_exp_full), max(y_flat)) + 0.5)
    ax.set_title(f"Zone: {zone}\nR² (poly): {best_r2:.4f} | exp: y = {a:.3f} * exp({b:.3f} * x)")
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature anomaly [°C]')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#animation update function
def update(frame):
    for i in range(len(zones)):
        x_anim = x_full_list[i][:frame]
        y_nn_anim = y_mlp_full_list[i][:frame]
        y_poly_anim = y_poly_full_list[i][:frame]
        y_exp_anim = y_exp_full_list[i][:frame]

        lines_nn[i].set_data(x_anim, y_nn_anim)
        lines_poly[i].set_data(x_anim, y_poly_anim)
        lines_exp[i].set_data(x_anim, y_exp_anim)

    return lines_nn + lines_poly + lines_exp

#launching animation
ani = animation.FuncAnimation(fig, update, frames=len(x_full_list[0]), interval=0.1, blit=True)

plt.show()

#static plot
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
axes2 = axes2.flatten()

for i in range(len(zones)):
    ax2 = axes2[i]

    #data from csv
    df_zone = df[df['zone'] == zones[i]]
    df_zone_full = df_zone[df_zone['type'].isin(['fit', 'forecast'])].copy()
    x_years = df_zone_full['year'].values.reshape(-1, 1).astype(np.float32)
    y_values = df_zone_full['temperature'].values.reshape(-1, 1).astype(np.float32)

    ax2.plot(x_years.flatten(), y_values.flatten(), label='Real data (fit + forecast)', color='lightgray', alpha=1.0)
    ax2.plot(x_full_list[i], y_mlp_full_list[i], label='Neural Network (MLP)', color='blue')
    ax2.plot(x_full_list[i], y_poly_full_list[i], label=f'Polynomial (deg. {best_degree})', color='red', linestyle=':')
    ax2.plot(x_full_list[i], y_exp_full_list[i], label='Exponential', color='green', linestyle='-.')

    ax2.set_xlim(1880, 2050)
    ax2.set_ylim(min(y_values.flatten()) - 0.5, max(max(y_mlp_full_list[i]), max(y_poly_full_list[i]), max(y_exp_full_list[i]), max(y_values.flatten())) + 0.5)
    ax2.set_title(f"Zone: {zones[i]}")
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Temperature anomaly [°C]')
    ax2.legend()
    ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
