Deep-Fuzzy Hybrid Indoor Air Quality (IAQ) Optimization
=======================================================

This repository contains a Python implementation of a real-time, edge-enabled hybrid framework for indoor air quality (IAQ) optimization, based on deep learning and fuzzy logic. The system collects data from indoor IoT sensors and external weather APIs, processes and forecasts comfort risk, and performs adaptive actuation using fuzzy logic.

Main Features:
--------------
- Data acquisition from IoT sensors (simulated) and weather APIs (OpenWeatherMap)
- Data preprocessing, feature engineering, and calculation of the Comfort Risk Index (CRI)
- BiLSTM-based machine learning for time-series forecasting
- Mamdani fuzzy logic for interpretable and adaptive control
- Visualization of IAQ metrics, predictions, and control surfaces

Usage Instructions:
-------------------
1. Clone the repository:
   git clone 

2. Install dependencies:
   pip install -r requirements.txt

3. API Key Setup:
   - The code uses the OpenWeatherMap API to retrieve real-time outdoor environmental data.
   - You must obtain a (free) API key from: https://openweathermap.org/api
   - Open the main script (`deep_fuzzy_iaq.py`).
   - Replace the placeholder with your actual API key:
       OWM_API_KEY = "your_openweathermap_key"
   - If you want to use AirVisual or any other API, add your key and modify the API request code accordingly.

4. Run the main pipeline:
   python deep_fuzzy_iaq (version 1.1 com.py).py

   - This will simulate data collection, perform all ML/fuzzy logic steps, and display/plot results.
   - Results are also saved to a CSV file (e.g., `results.csv`) for further analysis.

5. Run visualization separately:
   python visualize_iaq_results.py

   - This will load `results.csv` and generate time-series plots, heatmaps, and statistical visualizations.

API Notes:
----------
- **OpenWeatherMap:** Used for fetching outdoor temperature, humidity, pressure, and wind speed. Required for context-aware comfort index calculation.
- If you deploy with actual IoT sensors, you may wish to connect additional APIs or data sources. Just replace or extend the data acquisition functions in `deep_fuzzy_iaq.py`.

Customization:
--------------
- Swap the mock data generator for your real sensor data.
- Tune fuzzy logic rules and ML model parameters as needed.
- For MQTT/IoT device actuation, see the (commented) example code in the main script.

Contact:
--------
For questions or collaboration, please open an issue on the GitHub repository or contact the maintainer listed in the README.

---------------------------------------------
