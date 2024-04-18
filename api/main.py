"""
Datos de entrada del modelo:
["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
"concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", 
"smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", 
"radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
"concave points_worst", "symmetry_worst", "fractal_dimension_worst"]


{
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    "perimeter_mean": 87.46,
    "area_mean": 566.3,
    "smoothness_mean": 0.09779,
    "compactness_mean": 0.08129,
    "concavity_mean": 0.06664,
    "concave points_mean": 0.04781,
    "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699,
    "texture_se": 0.7886,
    "perimeter_se": 2.058,
    "area_se": 23.56,
    "smoothness_se": 0.008462,
    "compactness_se": 0.0146,
    "concavity_se": 0.02387,
    "concave points_se": 0.01315,
    "symmetry_se": 0.0198,
    "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11,
    "texture_worst": 19.26,
    "perimeter_worst": 99.7,
    "area_worst": 711.2,
    "smoothness_worst": 0.144,
    "compactness_worst": 0.1773,
    "concavity_worst": 0.239,
    "concave points_worst": 0.1288,
    "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259
}


"""
from fastapi import FastAPI, HTTPException, Body
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

# Definir modelo Pydantic para la validación de datos de entrada
class InputData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Cargar el modelo entrenado
model = joblib.load('/Users/Mi/Documents/Git/mlops/experiment_tracking/models/clasificador.pkl')

# Función para realizar la predicción
def breast_cancer_prediction(message: InputData):
    data = pd.DataFrame.from_records([message.dict()])
    label = model.predict(data)[0]
    return {'label': int(label)}

# Ruta principal de la API
@app.get('/')
def main():
    return {'message': 'Bienvenido a la API de breast cancer prediction'}

# Ruta para realizar la predicción
@app.post('/breast-cancer-prediction/')
def predict_breast_cancer(data: InputData):
    try:
        # Realizar la predicción
        model_pred = breast_cancer_prediction(data)
        return {'prediction': model_pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))