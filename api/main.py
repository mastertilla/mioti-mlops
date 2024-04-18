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
from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load('/Users/Mi/Documents/Git/mlops/experiment_tracking/models/clasificador.pkl')

app = FastAPI()


def breast_cancer_prediction(message: dict):
    # Data Prep
    data = pd.DataFrame.from_records([message])
    label = model.predict(data)[0]

    return {'label': int(label)}

@app.get('/')
def main():
    return {'message': 'Bienvenido a la API de breast cancer prediction'}

@app.post('/breast-cancer-prediction/')
def predict_heart_attack(message: dict):
    model_pred = breast_cancer_prediction(message)
    return {'prediction': model_pred}