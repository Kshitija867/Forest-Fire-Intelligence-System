# EcoGuard: Real-Time Forest Fire Risk Prediction Platform

EcoGuard is an end-to-end machine learning application that predicts the likelihood of forest fires using real-time weather data. The platform combines live weather information from the Open-Meteo API with calculated Canadian Fire Weather Index (FWI) features and a trained Random Forest model to generate localized fire risk predictions.

---

# System Architecture

The project is organized into independent modules, each responsible for a specific stage of the prediction pipeline.

### 1. Weather Data Collection (`weather_api.py`)

This module retrieves live weather data from the Open-Meteo REST API based on geographic coordinates. The following weather parameters are collected:

* Temperature (°C)
* Relative Humidity (%)
* Wind Speed (km/h)
* Rainfall (mm)

---

### 2. Fire Weather Index Calculation (`fire_indices.py`)

Using the live weather data, this module calculates important fire danger indices from the Canadian Forest Fire Weather Index (FWI) System.

The calculated features include:

* Fine Fuel Moisture Code (FFMC)
* Duff Moisture Code (DMC)
* Drought Code (DC)
* Initial Spread Index (ISI)
* Fire Weather Index (FWI)

These engineered features are used as inputs to the machine learning model.

---

### 3. Prediction Pipeline (`predictor.py`)

This module performs the complete inference process by:

* Loading the trained Random Forest model
* Loading the saved Standard Scaler
* Preparing the feature vector
* Scaling the input data
* Generating the fire risk prediction
* Returning prediction probabilities

---

### 4. User Interfaces

The project provides two ways to run predictions.

**Command-Line Interface (`predict_live.py`)**

Allows predictions directly from the terminal using latitude and longitude coordinates.

**Streamlit Dashboard (`app.py`)**

Provides an interactive web interface where users can:

* Select predefined forest locations
* Search locations using geocoding
* View live weather conditions
* View calculated Fire Weather Index values
* Receive real-time fire risk predictions

---

# Machine Learning Pipeline

The prediction workflow follows these steps:

1. Retrieve live weather data from Open-Meteo.
2. Calculate Fire Weather Index (FWI) features.
3. Prepare the complete feature vector.
4. Apply feature scaling using the saved Standard Scaler.
5. Generate predictions using the trained Random Forest model.
6. Display the predicted fire risk and model confidence.

---

# Project Structure

```text
Forest_Fire_Prediction/
│
├── models/
│   ├── forest_fire_rf_model.pkl
│   └── standard_scaler.pkl
│
├── src/
│   ├── weather_api.py
│   ├── fire_indices.py
│   ├── predictor.py
│   └── predict_live.py
│
├── app.py
└── README.md
```

---

# Model Performance

| Metric   | Value                    |
| -------- | ------------------------ |
| Model    | Random Forest Classifier |
| Accuracy | 97.96%                   |
| Task     | Binary Classification    |
| Output   | Fire Risk / No Fire Risk |

The model was trained using meteorological observations and Fire Weather Index features. During model selection, priority was given to reducing false negatives so that high-risk situations are less likely to be classified as safe.

---

# Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Geopy
* Requests
* Open-Meteo API

---

# Running the Project

### 1. Activate the virtual environment

```powershell
.\venv\Scripts\activate
```

---

### 2. Launch the Streamlit application

```powershell
streamlit run app.py
```

---

### 3. Run predictions from the command line

```powershell
python -m src.predict_live <latitude> <longitude>
```

Example:

```powershell
python -m src.predict_live 36.7525 3.0420
```

---

# Features

* Real-time weather data retrieval
* Automatic Fire Weather Index calculation
* Random Forest-based fire risk prediction
* Interactive Streamlit dashboard
* Geographic location search using geocoding
* Command-line prediction support
* Modular and maintainable project structure
