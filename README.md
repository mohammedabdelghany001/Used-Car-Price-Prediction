# 🚗 Used Car Price Prediction Platform (XGBoost)

An end-to-end Machine Learning web application that predicts the market value of used cars with high statistical precision. Built using a dataset of **over 400,000 vehicle sales records**, the core architecture leverages advanced data engineering and tree-based gradient boosting.

📊 **Production Model Accuracy ($R^2$ Score):** `87.65%`  
⏱️ **Average Inference Time:** `< 0.05 seconds`

---

## 🚀 Live Demo

The application is deployed and accessible via Streamlit Community Cloud:
👉 **[Insert Your Streamlit Live Link Here Later]**

---

## 🛠️ Key Features & Engineering Architecture

### 1. Robust Data Preprocessing

- **Outlier Isolation:** Leveraged the Interquartile Range ($IQR$) method to dynamically eliminate statistical anomalies in pricing and high-mileage attributes.
- **Text Standardization:** Global string normalization (lowercasing and stripping whitespace) to prevent discrete feature fragmentation.

### 2. Leakage-Free Encoding Pipeline

- **Target Encoding:** Applied to high-cardinality nominal values (`brand`, `model`) computed strictly within training folds to avoid data leakage, backed by a global mean fallback mechanism.
- **One-Hot Encoding:** Applied to low-cardinality features (`transmission`, `fuel_type`, `color`) with categorical dimension alignment.

### 3. Feature Engineering

- **`car_age`:** Derived from the temporal difference between production year and the current year.
- **`mileage_per_year`:** Captures the vehicle's annual utilization intensity.

---

## 📊 Model Benchmarking Results

| Model Architecture           | $R^2$ Score (Accuracy) | MAE (Avg Error) | Training Speed |
| :--------------------------- | :--------------------: | :-------------: | :------------: |
| **XGBoost Regressor (Base)** |       **87.65%**       |  **$1,597.16**  |   **1.063s**   |
| Random Forest (Base)         |         87.18%         |    $1,629.75    |     9.231s     |
| Linear Regression            |         70.37%         |    $2,511.77    |     0.122s     |

---

## 📦 Project Structure

```text
├── app.py                  # Streamlit Web Application Interface
├── requirements.txt         # Production Dependency Specifications
├── final_xgb_model.pkl      # Serialized Winning XGBoost Model
├── encoding_maps.pkl        # Serialized Target Encoding Fallbacks & Columns
└── notebooks/
    └── car_price_modeling.ipynb  # Comprehensive Data Cleaning & Training Pipeline
```
