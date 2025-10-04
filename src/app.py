import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src folder
MODEL_PATH = os.path.join(BASE_DIR, '..', 'CarPricePrediction_model.pkl')
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'car_data.csv')

# ----------------------------
# Load model safely
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# App UI
# ----------------------------
st.title("🚗 Car Selling Price Prediction with Insights")

st.write("Enter the details of the car to predict its selling price:")

year = st.number_input("Year of Purchase", min_value=1990, max_value=2024, value=2018, step=1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=100.0, value=6.5, step=0.1, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=42000, step=1000)
owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3], index=0)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"], index=0)
transmission = st.selectbox("Transmission", options=["Manual", "Automatic"], index=0)
seller_type = st.selectbox("Seller Type", options=["Dealer", "Individual"], index=0)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_input(year, present_price, kms_driven, owner, fuel_type, transmission, seller_type):
    data = {
        'Year': year,
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Owner': owner,
        'Fuel_Type_Diesel': 1 if fuel_type == "Diesel" else 0,
        'Fuel_Type_Petrol': 1 if fuel_type == "Petrol" else 0,
        'Transmission_Manual': 1 if transmission == "Manual" else 0,
        'Seller_Type_Individual': 1 if seller_type == "Individual" else 0
    }
    return pd.DataFrame([data])

# ----------------------------
# Dataset loading (default or upload)
# ----------------------------
st.sidebar.header("📂 Dataset Options")

uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded")
elif os.path.exists(DEFAULT_DATA_PATH):
    df = pd.read_csv(DEFAULT_DATA_PATH)
    st.sidebar.info("ℹ️ Using default dataset from /data/car_data.csv")
else:
    df = None
    st.sidebar.warning("⚠️ No dataset found. Upload a CSV to enable comparisons.")

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Selling Price"):
    if model is None:
        st.error("Model not loaded. Please check if `CarPricePrediction_model.pkl` exists.")
    else:
        input_df = preprocess_input(year, present_price, kms_driven, owner, fuel_type, transmission, seller_type)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Selling Price: ₹ {prediction:.2f} lakhs")

        # ----------------------------
        # Visualization Section
        # ----------------------------
        st.subheader("📊 Model Insights and Visualizations")

        # 1️⃣ Feature Importance
        try:
            feature_importance = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title("Feature Importance in Car Price Prediction")
            st.pyplot(fig)
        except Exception:
            st.info("Feature importance not available for this model type.")

        # 2️⃣ User Input vs Dataset averages
        if df is not None:
            avg_price = df["Selling_Price"].mean()
            avg_kms = df["Kms_Driven"].mean()
            avg_present = df["Present_Price"].mean()

            comparison = pd.DataFrame({
                'Metric': ['Predicted Price', 'Avg Selling Price',
                           'Your Present Price', 'Avg Present Price',
                           'Your KMs', 'Avg KMs'],
                'Value': [prediction, avg_price, present_price, avg_present, kms_driven, avg_kms]
            })

            fig2, ax2 = plt.subplots()
            sns.barplot(data=comparison, x='Metric', y='Value', palette='coolwarm', ax=ax2)
            plt.xticks(rotation=45)
            ax2.set_title("Your Car vs Dataset Averages")
            st.pyplot(fig2)

            # 3️⃣ Selling Price Distribution + Your Prediction
            fig3, ax3 = plt.subplots()
            sns.histplot(df["Selling_Price"], bins=20, kde=True, color="skyblue", ax=ax3)
            ax3.axvline(prediction, color="red", linestyle="--", linewidth=2, label=f"Your Prediction: ₹{prediction:.2f}L")
            ax3.set_title("Distribution of Selling Prices (Dataset vs Your Car)")
            ax3.legend()
            st.pyplot(fig3)

            # 4️⃣ Present Price vs Selling Price scatter (highlight your car)
            fig4, ax4 = plt.subplots()
            sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", color="gray", alpha=0.6, label="Dataset")
            ax4.scatter(present_price, prediction, color="red", s=100, label="Your Car")
            ax4.set_title("Your Car vs Dataset: Present Price vs Selling Price")
            ax4.legend()
            st.pyplot(fig4)

            # 5️⃣ Year-wise Selling Price trend + Your Car
            if "Year" in df.columns:
                yearly = df.groupby("Year")["Selling_Price"].mean().reset_index()
                fig5, ax5 = plt.subplots()
                sns.lineplot(data=yearly, x="Year", y="Selling_Price", marker="o", ax=ax5)
                ax5.scatter(year, prediction, color="red", s=100, label="Your Car")
                ax5.set_title("Average Selling Price by Year (Dataset vs Your Car)")
                ax5.legend()
                st.pyplot(fig5)

        else:
            st.warning("No dataset available for comparison plots.")
