import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Car Price Appraiser",
    page_icon="🚗",
    layout="centered"
)

# App Header and Description
st.title("🚗 Smart Car Price Appraiser & Market Scaler")
st.markdown("Enter the vehicle specifications below to calculate the fair market value based on the analysis of 400,000+ market listings.")
st.markdown("---")

# 2. Load Model and Encoding Pipeline with Caching for Performance
@st.cache_resource
def load_pipeline_files():
    with open('final_xgb_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('encoding_maps.pkl', 'rb') as f_maps:
        maps = pickle.load(f_maps)
    return model, maps

try:
    model, maps = load_pipeline_files()
    brand_means = maps['brand_means']
    model_means = maps['model_means']
    global_mean = maps['global_mean']
    final_columns = maps['final_columns']
except FileNotFoundError:
    st.error("❌ Critical Error: Model files (`final_xgb_model.pkl` or `encoding_maps.pkl`) not found in the current directory.")
    st.stop()

# 3. User Input Form Interface
with st.form("car_pricing_form"):
    st.subheader("📋 Vehicle Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fetch and sort available brands alphabetically
        available_brands = sorted(list(brand_means.keys()))
        selected_brand = st.selectbox("Select Car Brand:", available_brands)
        
        # Fetch and sort available models alphabetically
        available_models = sorted(list(model_means.keys()))
        selected_model = st.selectbox("Select Car Model:", available_models)

    with col2:
        # Vehicle Year and Mileage Inputs
        current_year = datetime.datetime.now().year
        selected_year = st.number_input("Manufacture Year:", min_value=1980, max_value=current_year, value=2020, step=1)
        kms_driven = st.number_input("Total Mileage (KMs):", min_value=0, max_value=1000000, value=50000, step=5000)

    st.markdown("---")
    st.subheader("⚙️ Technical Specs & Aesthetics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        transmission = st.selectbox("Transmission Type:", ["Automatic", "Manual"])
    with col4:
        fuel_type = st.selectbox("Fuel Type:", ["Petrol", "Diesel", "Hybrid", "Electric", "LPG"])
    with col5:
        color = st.selectbox("Exterior Color:", ["White", "Black", "Silver", "Gray", "Blue", "Red", "Other"])

    # Predict Button
    submit_button = st.form_submit_button("💰 Calculate Estimated Price")

# 4. Feature Engineering & Prediction Logic
if submit_button:
    # A) Real-time Feature Engineering (Matching Notebook logic)
    car_age = current_year - selected_year
    if car_age == 0:
        car_age = 0.5  # Prevent division by zero for brand new cars
        
    mileage_per_year = kms_driven / car_age
    
    # B) Mapping Categorical Values using Target Encoding Maps
    brand_encoded = brand_means.get(selected_brand, global_mean)
    model_encoded = model_means.get(selected_model, global_mean)
    
    # C) Constructing Input Dictionary
    input_data = {
        'year': selected_year,
        'kms_driven': kms_driven,
        'car_age': car_age,
        'mileage_per_year': mileage_per_year,
        'brand': brand_encoded,
        'model': model_encoded
    }
    
    # D) Structuring One-Hot Encoding to Match Training Columns Perfectly
    for col in final_columns:
        if col not in input_data:
            input_data[col] = 0
            
    # Setting binary flag (1) for selected categorical features
    if f"transmission_{transmission}" in input_data:
        input_data[f"transmission_{transmission}"] = 1
    if f"fuel_type_{fuel_type}" in input_data:
        input_data[f"fuel_type_{fuel_type}"] = 1
    if f"color_{color}" in input_data:
        input_data[f"color_{color}"] = 1
        
    # E) Creating Dataframe and Aligning Column Order
    input_df = pd.DataFrame([input_data])[final_columns]
    
    # F) Running Prediction via Saved XGBoost Model
    predicted_price = model.predict(input_df)[0]
    
    # G) Displaying Results Professionally
    st.success("🎉 Market valuation calculated successfully!")
    
    # Calculate fair range based on Model MAE (~$1,560)
    lower_range = max(500, predicted_price - 1560)
    upper_range = predicted_price + 1560
    
    st.metric(label="Estimated Market Price:", value=f"${predicted_price:,.2f}")
    st.info(f"💡 **Fair Market Range:** Valued between **${lower_range:,.2f}** and **${upper_range:,.2f}** depending on technical condition.")