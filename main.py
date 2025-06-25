# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Ciutat Vella Rental Price Predictor",
    description="API for predicting rental prices in Ciutat Vella, Barcelona",
    version="1.0.0"
)

# Define the path to your models folder
# This ensures it works both locally and on Render
MODEL_DIR = os.path.join(os.path.dirname(__file__), "Exported_models") # UPDATED PATH
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_ciutat_vella.pkl") 
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_ciutat_vella.pkl") 

# Load the trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Ensure your .pkl files are in the '{MODEL_DIR}' directory.")
    # For now, let's re-raise the error to make it clear if files are missing
    raise

# Define the input data structure using Pydantic
# This is crucial for FastAPI to validate incoming requests
class RentalFeatures(BaseModel):
    # These should match the features your model expects IN ORDER
    sq_meters: float
    sq_meters_built: int
    rooms: int
    bathrooms: int
    balcony: float
    terrace: float
    exterior: float
    floor: float
    elevator: float
    ac: float
    year_built: int
    dist_city_center: float
    dist_closest_station: float

# Define the prediction endpoint
@app.post("/predict")
async def predict_rental_price(features: RentalFeatures):
    """
    Predicts the rental price based on input features.
    """
    input_data_dict = features.model_dump() # Converts Pydantic model to a dictionary

    feature_names = [
        'sq_meters', 'sq_meters_built', 'rooms', 'bathrooms',
        'balcony', 'terrace', 'exterior', 'floor', 'elevator',
        'ac', 'year_built', 'dist_city_center', 'dist_closest_station'
    ]

    input_values = [input_data_dict[f] for f in feature_names]
    input_array = np.array([input_values])

    # Scale the input features
    scaled_features = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(scaled_features)[0] # [0] to get the single value

    # Return the prediction
    return {"estimated_rental_price": round(float(prediction), 2)} # Round and convert to float for JSON


# Optional: A simple root endpoint to check if the API is running
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Barcelona Rental Price Predictor API! Visit /docs for more."}