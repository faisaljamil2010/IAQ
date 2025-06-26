import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Load Results DataFrame
df = pd.read_csv("results.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# 2. Plot CRI and Pollutant Trends
plt.figure(figsize=(14,6))
plt.plot(df.index, df['CRI'], label='CRI', color='darkred')
plt.plot(df.index, df['CO2']/df['CO2'].max(), label='CO2 (normalized)')
plt.plot(df.index, df['PM2.5']/df['PM2.5'].max(), label='PM2.5 (normalized)')
plt.plot(df.index, df['humidity']/100, label='Humidity (normalized)')
plt.plot(df.index, df['temperature']/df['temperature'].max(), label='Temperature (normalized)')
plt.title('Comfort Risk Index and Indoor Pollutants (normalized)')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Plot Actual vs Predicted CRI (if available)
if "CRI_pred" in df.columns:
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["CRI"], label="Actual CRI", linewidth=2)
    plt.plot(df.index, df["CRI_pred"], label="Predicted CRI", linestyle='--')
    plt.title('Actual vs. Predicted Comfort Risk Index (CRI)')
    plt.xlabel('Time')
    plt.ylabel('CRI')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. Heatmap: Fan Speed (Fuzzy) Control Surface (if available)
if all(col in df.columns for col in ["CRI", "CO2", "fan_speed_fuzzy"]):
    sample = df.sample(1000) if len(df) > 1000 else df
    pivot = sample.pivot_table(index="CO2", columns="CRI", values="fan_speed_fuzzy", aggfunc='mean')
    plt.figure(figsize=(10,7))
    sns.heatmap(pivot, cmap="YlGnBu")
    plt.title('Fan Speed (Fuzzy) as function of CRI and CO2')
    plt.xlabel('CRI')
    plt.ylabel('CO2 (ppm)')
    plt.tight_layout()
    plt.show()

# 5. Correlation Matrix
plt.figure(figsize=(8,6))
corr = df[["CRI", "CO2", "PM2.5", "humidity", "temperature"]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: Environmental Variables and CRI')
plt.tight_layout()
plt.show()

# 6. Distribution Plots
plt.figure(figsize=(12,6))
sns.histplot(df['CRI'], bins=30, kde=True, color='crimson', label="CRI")
sns.histplot(df['CO2'], bins=30, kde=True, color='navy', label="CO2", alpha=0.5)
plt.title('Distribution of CRI and CO2')
plt.legend()
plt.tight_layout()
plt.show()
