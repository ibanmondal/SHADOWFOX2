
# Car Selling Price Prediction

## Overview
This project predicts the selling price of used cars using Python and Machine Learning techniques. It utilizes a Random Forest Regressor model trained on historical car data to provide accurate price estimates based on features like year, present price, kilometers driven, fuel type, transmission, and seller type.

## Project Structure
```
CarPricePrediction/
├── data/
│   └── car_data.csv                 # Dataset file
├── notebooks/
│   └── carpriceeda.ipynb            # Exploratory Data Analysis notebook
├── src/
│   ├── app.py                       # Streamlit web app for predictions
│   ├── predict_model.py             # Script for individual predictions
│   └── train_model.py               # Script for training the model
├── .gitignore                       # Git ignore file
├── CarPricePrediction_model.pkl     # Trained model file
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## Features
- Data preprocessing with one-hot encoding for categorical variables
- Model training using Random Forest Regressor
- Model evaluation with MAE, RMSE, and R² metrics
- Prediction script for individual car price estimation
- Exploratory Data Analysis (EDA) notebook for data insights
- Web-based prediction interface using Streamlit for real-time predictions

## Installation
1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```cmd
   cd CarPricePrediction
   ```
3. Create a virtual environment (optional but recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
4. Install the required dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Dataset
- Place your dataset file named `car_data.csv` in the `data/` folder.
- The dataset should include columns such as: Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner.
- Note: The 'Car_Name' column is dropped during preprocessing as it is not used in the model.

## Usage

### Training the Model
Run the training script to train the model and save it:
```cmd
python src/train_model.py
```
This will:
- Load and preprocess the data
- Train a Random Forest Regressor
- Evaluate the model on the test set
- Save the trained model as `CarPricePrediction_model.pkl`

### Making Predictions
Use the prediction script to predict the selling price for a new car:
```cmd
python src/predict_model.py
```
The script includes an example input. Modify the `data` DataFrame in the script to input your own car features.

### Exploratory Data Analysis
Open and run the Jupyter notebook for EDA:
```cmd
jupyter notebook notebooks/carpriceeda.ipynb
```
This notebook provides initial data exploration and visualizations.

### Running the Streamlit Web App
A simple web frontend is available to predict car prices in real-time.

To run the app locally:
```cmd
streamlit run src/app.py
```

This will launch a web interface where you can input car details and get instant price predictions.

## Model Details
- **Algorithm**: Random Forest Regressor
- **Hyperparameters**: n_estimators=100, random_state=42
- **Features**: Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Diesel, Fuel_Type_Petrol, Transmission_Manual, Seller_Type_Individual
- **Target**: Selling_Price

## Evaluation
The model is evaluated using the following metrics on the test set:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Example output after training:
```
MAE: [value]
RMSE: [value]
R²: [value]
Model saved to [path]
```

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- streamlit

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is open-source. Please check the license file for details.
