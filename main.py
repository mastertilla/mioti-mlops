"""
Scoring para predicción de la compra/no compra de un inmueble. Posibles valores:

- ingresos: los ingresos mensuales de la familia (int)
- gastos_comunes: pagos mensuales de luz, agua, gas, etc. (int)
- pago_coche: gastos mensuales asociados a uno o más coches, incluyendo combustible (int)
- gastos_otros: gastos mensuales en supermercado y otros elementos necesarios para vivir (int)
- ahorros: suma de ahorros disponibles para la compra de la casa (int)
- vivienda: precio de la vivienda que la familia desea comprar (int)
- estado_civil: estado civil del solicitante
  - 0: soltero
  - 1: casado
  - 2: divorciado
- hijos: cantidad de hijos menores que no trabajan (int)
- trabajo: tipo de empleo o situación laboral
  - 0: sin empleo
  - 1: autónomo (freelance)
  - 2: empleado
  - 3: empresario
  - 4: pareja de autónomos
  - 5: pareja de empleados
  - 6: pareja de autónomo y asalariado
  - 7: pareja de empresario y autónomo
  - 8: pareja de empresarios (o empresario y empleado)
  
 - Devuelve Compra (Si [1], No [0])

Datos de entrada del modelo:

['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros',
 'vivienda', 'estado_civil', 'hijos', 'trabajo']

No compra (0):

{
    "ingresos": 6745,
    "gastos_comunes": 944,
    "pago_coche": 123,
    "gastos_otros": 429,
    "ahorros": 43240,
    "vivienda": 636897,
    "estado_civil": 1,
    "hijos": 3,
    "trabajo": 6
}

Compra (1):

{
    "ingresos": 5000,
    "gastos_comunes": 800,
    "pago_coche": 100,
    "gastos_otros": 400,
    "ahorros": 30000,
    "vivienda": 100000,
    "estado_civil": 0,
    "hijos": 1,
    "trabajo": 3
}
"""

from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load('scoring_model.sav')
scaler = joblib.load('scaler.sav')

app = FastAPI()

def data_prep(message):
    df = pd.DataFrame([message])
    df_scaled = scaler.transform(df)
    
    return df_scaled

def scoring_prediction(message: dict):
    data = data_prep(message)
    label = model.predict(data)[0]
    
    return {'comprar': int(label)}

@app.get('/')
def main():
    return {'message': 'Buenas, este es el Scoring Model de Borja :)'}

@app.post('/scoring-prediction/')
def predict_scoring(message: dict):
    model_pred = scoring_prediction(message)
    return {'prediction': model_pred}