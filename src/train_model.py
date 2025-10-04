import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ----------------------------
# Paths (Windows & cross-platform safe)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src folder
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'car_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'CarPricePrediction_model.pkl')

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# Preprocessing
# ----------------------------
# Drop text column 'Car_Name' which is not numeric
if 'Car_Name' in df.columns:
    df = df.drop(['Car_Name'], axis=1)

# Convert categorical columns to numeric using one-hot encoding
categorical_cols = ['Fuel_Type','Transmission','Seller_Type']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Predict & Evaluate
# ----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# ----------------------------
# Save the trained model
# ----------------------------
pickle.dump(model, open(MODEL_PATH, 'wb'))
print(f"Model saved to {MODEL_PATH}")
