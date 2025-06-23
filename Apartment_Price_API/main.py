from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

# Load scaler and model
scaler = joblib.load("scaler_ciutat_vella.pkl")
model = joblib.load("modelo_ciutat_vella.pkl")

# Feature list (must match training order)
features = [
    'latitude', 'longitude', 'sq_meters', 'sq_meters_built', 'rooms',
    'bathrooms', 'balcony', 'terrace', 'exterior', 'floor',
    'elevator', 'ac', 'heating', 'year_built', 'quality',
    'dist_city_center', 'dist_closest_station', 'month'
]

# Define FastAPI app
app = FastAPI()

# Request schema
class ApartmentData(BaseModel):
    latitude: float
    longitude: float
    sq_meters: float
    sq_meters_built: float
    rooms: int
    bathrooms: int
    balcony: int
    terrace: int
    exterior: int
    floor: int
    elevator: int
    ac: int
    heating: int
    year_built: int
    quality: int
    dist_city_center: float
    dist_closest_station: float
    month: int

# Prediction endpoint
@app.post("/predict")
def predict_price(data: ApartmentData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[features]
    input_std = scaler.transform(input_df)
    prediction = model.predict(input_std)
    return {"estimated_price": round(float(prediction[0]), 2)}