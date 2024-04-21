from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import base64
from PIL import Image
import io

# Cargar el modelo
model = joblib.load('model.sav')

# Inicializar la aplicación FastAPI
app = FastAPI()

# Función para codificar el sexo
def sex_encoding(sex):
    return 0 if sex.lower() == 'male' else 1

# Función para preparar los datos
def data_prep(sex, age, weight, hour, bpm):
    sex_encoded = sex_encoding(sex)
    return pd.DataFrame([[sex_encoded, age, weight, hour, bpm]], columns=['sex', 'age', 'weight', 'hour', 'bpm'])

# Codificar imágenes como datos base64 y escalarlas
with open("sonriente.png", "rb") as f:
    sonriente_data = base64.b64encode(f.read()).decode("utf-8")
    sonriente_img = Image.open(f)
    sonriente_width, sonriente_height = sonriente_img.size
    sonriente_scaled_width = int(sonriente_width * 0.5)
    sonriente_scaled_height = int(sonriente_height * 0.5)
    sonriente_scaled_img = sonriente_img.resize((sonriente_scaled_width, sonriente_scaled_height))

with open("triste.jpg", "rb") as f:
    triste_data = base64.b64encode(f.read()).decode("utf-8")
    triste_img = Image.open(f)
    triste_width, triste_height = triste_img.size
    triste_scaled_width = int(triste_width * 0.5)
    triste_scaled_height = int(triste_height * 0.5)
    triste_scaled_img = triste_img.resize((triste_scaled_width, triste_scaled_height))

# Ruta principal
@app.get('/', response_class=HTMLResponse)
def main():
    form = """
    <html>
    <head>
    <title>API de Salud</title>
    </head>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
    <h1 style="color: #0066cc;">API de Salud</h1>
    <p>Por favor, ingresa tus datos para determinar tu estado de salud:</p>
    <form action="/predict-state/" method="post">
        <label for="sex">Sexo (male, female):</label><br>
        <input type="text" id="sex" name="sex"><br>
        <label for="age">Edad:</label><br>
        <input type="number" id="age" name="age"><br>
        <label for="weight">Peso:</label><br>
        <input type="number" id="weight" name="weight"><br>
        <label for="hour">Hora del día (0-23):</label><br>
        <input type="number" id="hour" name="hour"><br>
        <label for="bpm">Pulsaciones por minuto:</label><br>
        <input type="number" id="bpm" name="bpm"><br><br>
        <input type="submit" value="Predecir estado de salud">
    </form>
    </body>
    </html>
    """
    return form

# Ruta para predecir el estado de salud
@app.post('/predict-state/', response_class=HTMLResponse)
async def predict_state(sex: str = Form(..., description="Sexo (male, female)"),
                        age: int = Form(..., description="Edad"),
                        weight: float = Form(..., description="Peso"),
                        hour: int = Form(..., description="Hora del día (0-23)"),
                        bpm: int = Form(..., description="Pulsaciones por minuto")):

    try:
        # Validar los datos de entrada
        if sex.lower() not in ['male', 'female']:
            raise ValueError("Sexo debe ser 'male' o 'female'")

        if not (0 <= hour <= 23):
            raise ValueError("La hora debe estar entre 0 y 23")

        # Preparar los datos
        data = data_prep(sex, age, weight, hour, bpm)

        # Hacer la predicción
        prediction = model.predict(data)[0]

        # Determinar el estado de salud
        health_state = "bien" if prediction == 1 else "mal"

        # Formatear el mensaje de estado de salud en HTML con imágenes incrustadas y escaladas
        if health_state == "bien":
            message = "<h2>Tu estado de salud es bueno</h2>"
            sonriente_scaled_data_bytes = io.BytesIO()
            sonriente_scaled_img.save(sonriente_scaled_data_bytes, format='PNG')
            sonriente_scaled_data_bytes = sonriente_scaled_data_bytes.getvalue()
            sonriente_scaled_data_encoded = base64.b64encode(sonriente_scaled_data_bytes).decode("utf-8")
            image = f'<img src="data:image/png;base64,{sonriente_scaled_data_encoded}" alt="Imagen de sonrisa" width="{sonriente_scaled_width}px">'
        else:
            message = "<h2>Tu estado de salud es malo</h2>"
            triste_scaled_data_bytes = io.BytesIO()
            triste_scaled_img.save(triste_scaled_data_bytes, format='JPEG')
            triste_scaled_data_bytes = triste_scaled_data_bytes.getvalue()
            triste_scaled_data_encoded = base64.b64encode(triste_scaled_data_bytes).decode("utf-8")
            image = f'<img src="data:image/jpg;base64,{triste_scaled_data_encoded}" alt="Imagen de tristeza" width="{triste_scaled_width}px">'

        # Combinar el mensaje y la imagen en una sola respuesta HTML
        response_html = f"{message}{image}"

        return response_html

    except ValueError as ve:
        error_message = f"<h2>Error:</h2><p>{ve}</p>"
        return error_message

