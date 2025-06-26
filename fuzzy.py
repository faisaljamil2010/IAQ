import numpy as np
import matplotlib.pyplot as plt

def trimf(x, a, b, c):
    return np.clip(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0, 1)

def plot_with_labels(ax, x, sets, title, xlabel):
    for label, (a, b, c) in sets.items():
        ax.plot(x, trimf(x, a, b, c), color='black')
        ax.text(b, 0.9, label, ha='center', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Membership')
    ax.set_ylim(0.2, 1.05)
    ax.grid(False)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs = axs.flatten()

# CRI (0 to 1)
x_cri = np.linspace(0, 1.1, 500)
cri_sets = {
    'Low': (0.0, 0.2, 0.4),
    'Moderate': (0.3, 0.5, 0.7),
    'High': (0.6, 0.8, 1.0)
}
plot_with_labels(axs[0], x_cri, cri_sets, 'Comfort Risk Index (CRI)', 'CRI')

# PM2.5 (0 to 150 µg/m³)
x_pm = np.linspace(0, 150, 500)
pm_sets = {
    'Good': (0, 30, 60),
    'Moderate': (50, 90, 120),
    'Poor': (110, 130, 150)
}
plot_with_labels(axs[1], x_pm, pm_sets, 'PM$_{2.5}$ (µg/m³)', 'PM$_{2.5}$')

# CO₂ (400 to 2000 ppm)
x_co2 = np.linspace(400, 2000, 500)
co2_sets = {
    'Normal': (400, 700, 1000),
    'Elevated': (900, 1200, 1500),
    'Unhealthy': (1400, 1700, 2000)
}
plot_with_labels(axs[2], x_co2, co2_sets, 'CO$_2$ (ppm)', 'CO$_2$ (ppm)')

# Fan Speed (Output)
x_fan = np.linspace(1, 3.2, 500)
fan_sets = {
    'Low': (1.0, 1.5, 2.0),
    'Medium': (1.8, 2.0, 2.2),
    'High': (2.5, 2.8, 3.0)
}
plot_with_labels(axs[3], x_fan, fan_sets, 'Fan Speed (Output)', 'Fan Speed')

plt.tight_layout()
plt.show()
