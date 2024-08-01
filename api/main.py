"""
Datos de entrada del modelo:
['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'gender_Other', 'ever_married_Yes',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes']

{
    'age': int,
    'hypertension': int (1/0),
    'gender': str (male/female/other),
    'ever_married_Yes': int (1/0),
    'heart_disease': int (1/0),
    'avg_glucose_level': int,
    'bmi': int,
    'work_type': str (never worked/private/self-employed/children)
    'residence_type': str (urban)
    'smoking_status': str (formerly smoked/never smoked/smokes)
}

{
    "age": 33,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 0,
    "avg_glucose_level": 70,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

{
    "age": 75,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 1,
    "avg_glucose_level": 120,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

"""
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, conint, confloat
import joblib
import pandas as pd
from typing import Literal

# Cargar el modelo previamente entrenado
model = joblib.load('model.sav')

app = FastAPI()

# Definición del DataModel utilizando Pydantic
class PatientData(BaseModel):
    age: conint(ge=0)  # Edad debe ser un número entero positivo
    hypertension: conint(ge=0, le=1)  # 0 o 1
    heart_disease: conint(ge=0, le=1)  # 0 o 1
    avg_glucose_level: confloat(ge=0)  # Nivel de glucosa promedio debe ser un número positivo
    bmi: confloat(ge=0)  # BMI debe ser un número positivo
    gender: Literal['male', 'female', 'other']  # Valores permitidos
    ever_married_Yes: conint(ge=0, le=1)  # 0 o 1
    work_type: Literal['never worked', 'private', 'self-employed', 'children']  # Valores permitidos
    residence_type: Literal['urban', 'rural']  # Valores permitidos
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes']  # Valores permitidos

# Definición de la clave API
API_KEY = "Contrasena"

def api_key_dependency(api_key: str = Query(..., alias="api_key")):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Funciones de encoding
def gender_encoding(message):
    gender_encoded = {'gender_Male': 0, 'gender_Other': 0}
    if message['gender'].lower() == 'male':
        gender_encoded['gender_Male'] = 1
    elif message['gender'].lower() == 'other':
        gender_encoded['gender_Other'] = 1

    del message['gender']
    message.update(gender_encoded)

def work_type_encoding(message):
    work_type_encoded = {'work_type_Never_worked': 0, 'work_type_Private': 0,
                         'work_type_Self-employed': 0, 'work_type_children': 0}

    if message['work_type'].lower() == 'never worked':
        work_type_encoded['work_type_Never_worked'] = 1
    elif message['work_type'].lower() == 'private':
        work_type_encoded['work_type_Private'] = 1
    elif message['work_type'].lower() == 'self-employed':
        work_type_encoded['work_type_Self-employed'] = 1
    elif message['work_type'].lower() == 'children':
        work_type_encoded['work_type_children'] = 1

    del message['work_type']
    message.update(work_type_encoded)

def residence_encoding(message):
    residence_encoded = {'Residence_type_Urban': 0}
    if message['residence_type'] == 'urban':
        residence_encoded['Residence_type_Urban'] = 1

    del message['residence_type']
    message.update(residence_encoded)

def smoking_encoding(message):
    smoking_encoded = {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 0,
                       'smoking_status_smokes': 0}
    if message['smoking_status'] == 'formerly smoked':
        smoking_encoded['smoking_status_formerly smoked'] = 1
    elif message['smoking_status'] == 'never smoked':
        smoking_encoded['smoking_status_never smoked'] = 1
    elif message['smoking_status'] == 'smokes':
        smoking_encoded['smoking_status_smokes'] = 1

    del message['smoking_status']
    message.update(smoking_encoded)

def data_prep(message):
    gender_encoding(message)
    work_type_encoding(message)
    residence_encoding(message)
    smoking_encoding(message)
    return pd.DataFrame([message])

def heart_prediction(message: PatientData):
    # Convertir el objeto de entrada a diccionario
    data_dict = message.dict()
    # Preparar los datos
    data = data_prep(data_dict)
    # Realizar la predicción
    label = model.predict(data)[0]
    return {'label': int(label)}

@app.get('/')
def main():
    return {'message': 'Hola'}

@app.post('/heart-attack-prediction/')
def predict_heart_attack(message: PatientData, api_key: str = Depends(api_key_dependency)):
    model_pred = heart_prediction(message)
    return {'prediction': model_pred}
