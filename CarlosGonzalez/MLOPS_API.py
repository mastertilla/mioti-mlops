# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# %%
# Carga del modelo
model = joblib.load(r'C:\Users\carlo\OneDrive\Escritorio\ML OPS\Api trabajo\Regresion_mlops.pkl')

# %%
# Modelo Pydantic para validación de datos
class DatosHipoteca(BaseModel):
    ingresos: int = Field(..., description="Ingresos mensuales de la familia")
    gastos_comunes: int = Field(..., description="Pagos mensuales de servicios básicos")
    pago_coche: int = Field(..., description="Pagos mensuales por coche y gastos en combustible")
    gastos_otros: int = Field(..., description="Gastos mensuales en supermercado y otros necesarios para vivir")
    ahorros: int = Field(..., description="Suma de ahorros para la compra de la casa")
    vivienda: int = Field(..., description="Precio de la vivienda que desea comprar")
    estado_civil: int = Field(..., description="Estado civil del solicitante", ge=0, le=2)
    hijos: int = Field(..., description="Cantidad de hijos menores que no trabajan")
    trabajo: int = Field(..., description="Tipo de empleo del solicitante", ge=0, le=8)

# %%
# Configuración de FastAPI y SlowAPI para limitación de tasa
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get('/')
@limiter.limit("5/minute")
async def main(request: Request):
    return {'message': 'Bienvenido a la API de predicción de hipotecas'}

@app.post('/decision/')
@limiter.limit("5/minute")
async def decision_final(request: Request,datos: DatosHipoteca):
    datos_df = pd.DataFrame([datos.dict()])
    prediccion = model.predict(datos_df)[0]
    return {'prediccion': int(prediccion)}


# %%



