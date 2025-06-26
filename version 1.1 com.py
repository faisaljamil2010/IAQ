# deep_fuzzy_iaq.py
import pandas as pd
import numpy as np
import requests
import datetime
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# === 1. API & Sensor Data Acquisition ===
OWM_API_KEY = "your_openweathermap_key"

def get_indoor_sensor_data():
    now = datetime.datetime.utcnow()
    return {
        "timestamp": now,
        "CO2": np.random.uniform(400, 2000),
        "PM2.5": np.random.uniform(5, 150),
        "humidity": np.random.uniform(30, 80),
        "temperature": np.random.uniform(18, 30)
    }

def get_external_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={OWM_API_KEY}&units=metric"
    r = requests.get(url)
    data = r.json()
    return {
        "AQI": np.random.randint(50, 150),
        "temperature": data['main']['temp'],
        "humidity": data['main']['humidity'],
        "pressure": data['main']['pressure'],
        "wind_speed": data['wind']['speed']
    }

# Simulate data collection
indoor_data = [get_indoor_sensor_data() for _ in range(24*60)]
df_indoor = pd.DataFrame(indoor_data)
df_indoor["timestamp"] = pd.to_datetime(df_indoor["timestamp"])

# === 2. Preprocessing & Feature Engineering ===
df_indoor.set_index("timestamp", inplace=True)
df = df_indoor.resample('1T').mean().interpolate()

for col in ["CO2", "PM2.5", "humidity", "temperature"]:
    df[f"{col}_mean10"] = df[col].rolling(10, min_periods=1).mean()
    df[f"{col}_std10"] = df[col].rolling(10, min_periods=1).std()
    df[f"{col}_lag1"] = df[col].shift(1)

df["hour"] = df.index.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"]/24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"]/24)
df.dropna(inplace=True)

# === 3. Comfort Risk Index (CRI) Calculation ===
def compute_cri(row, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, eta=0.1):
    xCO2 = (row["CO2"] - 400) / (2000 - 400)
    xPM = row["PM2.5"] / 150
    xH = row["humidity"] / 100
    D_bar = 0.5
    I = xPM * xH
    z = alpha*xCO2 + beta*xPM + gamma*xH + delta*D_bar + eta*I
    cri = 1 / (1 + np.exp(-z))
    return cri

df["CRI"] = df.apply(compute_cri, axis=1)

# === 4. Machine Learning: BiLSTM Student Model ===
scaler = MinMaxScaler()
features = ["CO2", "PM2.5", "humidity", "temperature", "hour_sin", "hour_cos"]
target = "CRI"
X = scaler.fit_transform(df[features])
y = df[target].values

def create_sequences(X, y, seq_length=10):
    xs, ys = [], []
    for i in range(len(X)-seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

input_dim = X_train.shape[2]
hidden_dim = 32
output_dim = 1
model = BiLSTM(input_dim, hidden_dim, output_dim)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Predict
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

# === 5. Fuzzy Logic Control (Mamdani) ===
cri_fz = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cri')
co2_fz = ctrl.Antecedent(np.arange(400, 2001, 1), 'co2')
pm25_fz = ctrl.Antecedent(np.arange(0, 151, 1), 'pm25')
fan_speed = ctrl.Consequent(np.arange(0, 3, 1), 'fan_speed')

cri_fz['low'] = fuzz.trimf(cri_fz.universe, [0, 0, 0.4])
cri_fz['moderate'] = fuzz.trimf(cri_fz.universe, [0.2, 0.5, 0.8])
cri_fz['high'] = fuzz.trimf(cri_fz.universe, [0.6, 1, 1])
co2_fz['normal'] = fuzz.trimf(co2_fz.universe, [400, 400, 1000])
co2_fz['elevated'] = fuzz.trimf(co2_fz.universe, [800, 1200, 1600])
co2_fz['unhealthy'] = fuzz.trimf(co2_fz.universe, [1500, 2000, 2000])
pm25_fz['good'] = fuzz.trimf(pm25_fz.universe, [0, 0, 30])
pm25_fz['moderate'] = fuzz.trimf(pm25_fz.universe, [20, 60, 100])
pm25_fz['poor'] = fuzz.trimf(pm25_fz.universe, [80, 150, 150])
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 1])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [0, 1, 2])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [1, 2, 2])

rule1 = ctrl.Rule(cri_fz['high'] & co2_fz['unhealthy'] & pm25_fz['poor'], fan_speed['high'])
rule2 = ctrl.Rule(cri_fz['moderate'] & co2_fz['elevated'], fan_speed['medium'])
rule3 = ctrl.Rule(cri_fz['low'] & co2_fz['normal'] & pm25_fz['good'], fan_speed['low'])

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

last_row = df.iloc[-1]
fan_sim.input['cri'] = last_row['CRI']
fan_sim.input['co2'] = last_row['CO2']
fan_sim.input['pm25'] = last_row['PM2.5']
fan_sim.compute()
print(f"Recommended fan speed: {fan_sim.output['fan_speed']}")

# === 6. Visualization ===
plt.figure(figsize=(12,5))
plt.plot(df.index, df['CRI'], label='CRI')
plt.plot(df.index, df['CO2']/df['CO2'].max(), label='CO2 (norm)')
plt.plot(df.index, df['PM2.5']/df['PM2.5'].max(), label='PM2.5 (norm)')
plt.legend()
plt.title('Comfort Risk Index & Pollutants')
plt.show()

cri_grid, co2_grid = np.meshgrid(np.linspace(0,1,50), np.linspace(400,2000,50))
fan_result = np.zeros_like(cri_grid)
for i in range(fan_result.shape[0]):
    for j in range(fan_result.shape[1]):
        fan_sim.input['cri'] = cri_grid[i, j]
        fan_sim.input['co2'] = co2_grid[i, j]
        fan_sim.input['pm25'] = 75
        fan_sim.compute()
        fan_result[i, j] = fan_sim.output['fan_speed']
plt.figure(figsize=(8,6))
plt.contourf(cri_grid, co2_grid, fan_result, cmap='viridis')
plt.xlabel('CRI')
plt.ylabel('CO2')
plt.title('Fan speed control surface')
plt.colorbar(label='Fan Speed')
plt.show()

# === 7. (Optional) IoT Actuation: MQTT Example (commented) ===
# import paho.mqtt.client as mqtt
# client = mqtt.Client()
# client.connect("broker.hivemq.com", 1883, 60)
# client.publish("iaq/fan", str(fan_sim.output['fan_speed']))
