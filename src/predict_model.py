import os
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'CarPricePrediction_model.pkl')

# Load model
model = pickle.load(open(MODEL_PATH, 'rb'))
print("Model loaded successfully!")

# Example input
data = pd.DataFrame({
    'Year': [2020],
    'Present_Price': [5.5],      # Current showroom price in lakhs
    'Kms_Driven': [38000],
    'Owner': [2],
    'Fuel_Type_Diesel': [0],
    'Fuel_Type_Petrol': [1],
    'Transmission_Manual': [1],
    'Seller_Type_Individual': [0]
})

# Align columns with trained model
trained_columns = model.feature_names_in_
data = data.reindex(columns=trained_columns, fill_value=0)

# Predict
predicted_price = model.predict(data)
print(f"Predicted Selling Price: {predicted_price[0]:.2f}")
