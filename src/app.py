import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'CarPricePrediction_model.pkl')

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    return model

model = load_model()

st.title("Car Selling Price Prediction")

st.write("Enter the details of the car to predict its selling price:")

# Input fields
year = st.number_input("Year of Purchase", min_value=1990, max_value=2024, value=2018, step=1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=100.0, value=6.5, step=0.1, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=42000, step=1000)
owner = st.selectbox("Number of Previous Owners", options=[0,1,2,3], index=0)

fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"], index=0)
transmission = st.selectbox("Transmission", options=["Manual", "Automatic"], index=0)
seller_type = st.selectbox("Seller Type", options=["Dealer", "Individual"], index=0)

# Preprocess inputs to match model features
def preprocess_input(year, present_price, kms_driven, owner, fuel_type, transmission, seller_type):
    current_year = 2024
    car_age = current_year - year

    data = {
        'Year': year,
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Owner': owner,
        # One-hot encoding for Fuel_Type
        'Fuel_Type_Diesel': 1 if fuel_type == "Diesel" else 0,
        'Fuel_Type_Petrol': 1 if fuel_type == "Petrol" else 0,
        # One-hot encoding for Transmission
        'Transmission_Manual': 1 if transmission == "Manual" else 0,
        # One-hot encoding for Seller_Type
        'Seller_Type_Individual': 1 if seller_type == "Individual" else 0
    }
    return pd.DataFrame([data])

input_df = preprocess_input(year, present_price, kms_driven, owner, fuel_type, transmission, seller_type)

if st.button("Predict Selling Price"):
    # Align columns with model features
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Selling Price: ₹ {prediction:.2f} lakhs")
