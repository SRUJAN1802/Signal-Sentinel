# 📡 Signal Sentinel

Signal Sentinel is an AI-powered telecom network monitoring system that predicts network health using machine learning.

The project analyzes signal strength, throughput, latency, network type, and radio measurements to generate a network health score, identify potential issues, and provide recommendations for improving network performance.

## Features

* Network Health Prediction
* Risk Score Calculation
* Root Cause Analysis
* AI Recommendations
* Network Stability Index
* Link Quality Score
* Failure Probability Estimation
* Radio Link Forecast
* Prediction History
* PDF Report Generation
* Interactive Visualizations

## Technologies Used

* Python
* Streamlit
* XGBoost
* Pandas
* NumPy
* Plotly
* Scikit-Learn

## Project Structure

* `app.py` – Main Streamlit application
* `model.pkl` – Trained machine learning model
* `network_encoder.pkl` – Network type encoder
* `feature_columns.pkl` – Feature list used during prediction
* `cleaned_network_data.csv` – Dataset
* `feature_importance.csv` – Model feature importance values

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Purpose

This project was developed to explore the use of machine learning for telecom network monitoring and radio link degradation analysis. It provides an interactive dashboard for analyzing network conditions and generating actionable insights.

## Author

Srujan
