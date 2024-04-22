"""
Modelo predicción de precios ['muy bajo' < 'bajo' < 'medio' < 'alto' < 'muy alto'] 
de pisos por características.
Datos de entrada del modelo (características):
['review_scores_rating', 'room_type', 'bedrooms', 'bathrooms' ]

{
'review_scores_rating': int (0-100), 
'room_type': str (Entire home/apt, Private room, Shared room),
'bedrooms' float, 
'bathrooms': float,
}

{
'review_scores_rating': 80, 
'room_type': str Shared room,
'bedrooms' 2.0, 
'bathrooms': 1.0,
}

{
'review_scores_rating': 70, 
'room_type': str Private room,
'bedrooms' 3.0, 
'bathrooms': 2.0,
}

{
'review_scores_rating': 50, 
'room_type': str Private room,
'bedrooms' 6.0, 
'bathrooms': 4.0,
}

"""
from fastapi import FastAPI, HTTPException, Depends, Security
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

#Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Modelo de datos de entrada con Pydantic
class InputData(BaseModel):
    review_scores_rating: int
    room_type: str
    bedrooms: float
    bathrooms: float

model = joblib.load('random_forest_model.sav')
app = FastAPI()

#Función para ETL + Predicción
def predict_price(data: InputData):
    df = pd.DataFrame(data.dict(), index=[0])
    
    #Aplicamos one-hot encoding a la columna 'room_type' como en el entrenamiento
    df = pd.get_dummies(df, columns=['room_type'])
    room_type_columns = ['room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room']
    for col in room_type_columns:
        if col not in df.columns:
            df[col] = 0

    columns = ['review_scores_rating', 'bedrooms', 'bathrooms'] + room_type_columns
    X = df[columns]
    prediction = model.predict(X)
    return prediction[0]

#Función autentificación por token 'MI_TOKEN' + Error 401
def authenticate(auth: str):
    if auth != 'MI_TOKEN':
        raise HTTPException(status_code=401, detail='Token de autenticación no válido')
    return True

#Función validar datos + Predicción + Error 400/500
@app.post('/predict/')
def predict(data: InputData, auth: bool = Security(authenticate, scopes=['predict'])):
    try:
        #Validamos los datos de entrada o devolvemos error
        if data.room_type not in ['Entire home/apt', 'Private room', 'Shared room']:
            raise HTTPException(status_code=400, detail="Invalid room_type")
        prediction = predict_price(data)
        return {'Prediction': prediction}
    
    #Error de predicción 500
    except Exception as e:
        logger.error(f'Error durante la predicción: {e}')
        raise HTTPException(status_code=500, detail='Error durante la predicción')

