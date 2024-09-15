from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np

try:
    model = joblib.load('precio_model.sav')
    print("Modelo cargado correctamente")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

app = FastAPI()

def validate_bedrooms(bedrooms):
    if bedrooms < 0 or bedrooms > 10:
        raise HTTPException(status_code=400, detail="El número de habitaciones debe estar entre 0 y 10.")
    return bedrooms

def validate_bathrooms(bathrooms):
    if bathrooms < 0 or bathrooms > 8:
        raise HTTPException(status_code=400, detail="El número de baños debe estar entre 0 y 8.")
    return bathrooms

def validate_review_score(score):
    if score < 0 or score > 100:
        raise HTTPException(status_code=400, detail="La puntuación de la reseña debe estar entre 0 y 100.")
    return score

def property_type_encoding(message):
    message['property_type'] = message['property_type'].strip().title()
    
    valid_property_types = [
        'House', 'Condominium', 'Bed & Breakfast', 'Loft',
        'Boat', 'Boutique hotel', 'Bungalow', 'Camper/RV', 'Casa particular',
        'Chalet', 'Dorm', 'Earth House', 'Guest suite', 'Guesthouse', 'Hostel',
        'Other', 'Serviced apartment', 'Tent', 'Timeshare', 'Townhouse', 'Villa'
    ]

    if message['property_type'] not in valid_property_types:
        print(f"Advertencia: Valor inválido '{message['property_type']}' en 'property_type'. Se asignará 'Other'.")
        message['property_type'] = 'Other'

    property_type_encoded = {f"property_type_{ptype}": 0 for ptype in valid_property_types}
    property_type_key = f"property_type_{message.get('property_type', 'Other')}"
    
    if property_type_key in property_type_encoded:
        property_type_encoded[property_type_key] = 1
    else:
        property_type_encoded['property_type_Other'] = 1

    message.pop('property_type', None)
    message.update(property_type_encoded)

def room_type_encoding(message):
    room_type_encoded = {'room_type_Entire home/apt': 0, 'room_type_Private room': 0, 'room_type_Shared room': 0}
    
    if message['room_type'] == 'Entire home/apt':
        room_type_encoded['room_type_Entire home/apt'] = 1
    elif message['room_type'] == 'Private room':
        room_type_encoded['room_type_Private room'] = 1
    elif message['room_type'] == 'Shared room':
        room_type_encoded['room_type_Shared room'] = 1

    del message['room_type']
    message.update(room_type_encoded)

def data_prep(message):
    message['bedrooms'] = validate_bedrooms(message.get('bedrooms', 0))
    message['bathrooms'] = validate_bathrooms(message.get('bathrooms', 0))
    message['review_scores_rating'] = validate_review_score(message.get('review_scores_rating', 0))
    
    property_type_encoding(message)
    room_type_encoding(message)

    data = pd.DataFrame(message, index=[0])
    
    expected_columns = [
        'bedrooms', 'bathrooms', 'review_scores_rating',
        'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
        'property_type_Bed & Breakfast', 'property_type_Boat', 'property_type_Boutique hotel',
        'property_type_Bungalow', 'property_type_Camper/RV', 'property_type_Casa particular',
        'property_type_Chalet', 'property_type_Condominium', 'property_type_Dorm',
        'property_type_Earth House', 'property_type_Guest suite', 'property_type_Guesthouse',
        'property_type_Hostel', 'property_type_House', 'property_type_Loft',
        'property_type_Other', 'property_type_Serviced apartment', 'property_type_Tent',
        'property_type_Timeshare', 'property_type_Townhouse', 'property_type_Villa'
    ]
    
    expected_columns = list(model.feature_names_in_)

    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_columns]
    print("Columnas actuales en los datos:", data.columns)
    return data

def price_prediction(message: dict):
    try:
        data = data_prep(message)
        label = model.predict(data)[0]
        return {'label': int(label)}
    except Exception as e:
        return {'error': str(e)}

class PropertyData(BaseModel):
    bedrooms: int
    bathrooms: int
    review_scores_rating: int
    property_type: str
    room_type: str

@app.get('/')
def main():
    return {'message': 'API de predicción de precios de propiedades'}

@app.post("/predict/")
async def predict_price(property_data: PropertyData):
    try:
        message = property_data.dict()
        result = price_prediction(message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
