from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# Cargar el modelo guardado
model = joblib.load('model.sav')

app = FastAPI()


# Definir un modelo Pydantic para la entrada
class DiabetesInput(BaseModel):
    age: float
    sex: str
    bmi: float
    bp: float
    cholesterol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    class Config:
        schema_extra = {
            "example": {
                "age": 50,
                "sex": "male",
                "bmi": 25.0,
                "bp": 80.0,
                "cholesterol": 200.0,
                "fbs": 100.0,
                "restecg": 0.0,
                "thalach": 150.0,
                "exang": 0.0,
                "oldpeak": 1.5
            }
        }

def encode_data(data: DiabetesInput):
    # Crear un diccionario de datos
    data_dict = data.dict()

    # Realizar el encoding manualmente
    data_dict['sex'] = 1 if data_dict['sex'].lower() == 'male' else 0

    return pd.DataFrame([data_dict])


@app.get('/')
def main():
    return {'message': 'Ve a este mismo sitio /docs para poder probar valores'}


@app.post('/diabetes-prediction/')
def predict_diabetes(data: DiabetesInput):
    df = encode_data(data)
    prediction = model.predict(df)[0]
    return {'prediction': int(prediction)}
