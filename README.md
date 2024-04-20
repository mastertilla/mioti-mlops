# Breast Cancer Prediction API con MLflow

Esta API proporciona un servicio para predecir el cáncer de mama basado en un modelo de aprendizaje automático entrenado. Utiliza un modelo de clasificación RandomForest para realizar predicciones sobre datos de entrada proporcionados por el usuario.

## Requisitos

- Python 3.6 o superior
- Bibliotecas Python especificadas en `requirements.txt`

## Instalación

1. Clona este repositorio:

```
git clone https://github.com/tu_usuario/tu_repositorio.git
```

2. Navega hasta el directorio del proyecto:

```
cd tu_proyecto
```

3. Instala las dependencias:

```
pip install -r requirements.txt
```

## Uso

El primer paso es lanzar MLFlow para poder guardar experimentos en local.

cd experiment_tracking

mlflow server --port 5000

Esto hará que MLFlow esté disponible en esta url: http://127.0.0.1:5000. 

### 1. Entrenamiento del Modelo con MLflow

Para entrenar el modelo y realizar un seguimiento de los experimentos con MLflow, ejecuta el script `modelo_hyper_mlflow.py`. Este script realiza el preprocesamiento de los datos, entrena el modelo y guarda los resultados en MLflow.

```bash
python modelo_hyper_mlflow.py
```

Luego, abre `http://localhost:5000` en tu navegador para ver el tablero de MLflow.

### 2. Iniciar la API

Para iniciar la API, ejecuta el script `main.py`. La API estará disponible en `http://localhost:8000`.

```bash
uvicorn main:app --reload
```

### 3. Obtener un Token de Acceso

Antes de realizar predicciones, necesitas obtener un token de acceso. Envía una solicitud POST a `/token` con las credenciales de usuario (`username` y `password`) en el cuerpo de la solicitud.

```bash
curl -X POST "http://localhost:8000/token" -d "username=testuser&password=testpassword"
```

### 4. Realizar una Predicción

Una vez que tengas un token de acceso, puedes realizar predicciones enviando una solicitud POST a `/breast-cancer-prediction/` con los datos de entrada en formato JSON y el token de acceso en el encabezado de autorización.

```bash
curl -X POST "http://localhost:8000/breast-cancer-prediction/" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
           "radius_mean": 13.54,
           "texture_mean": 14.36,
           "perimeter_mean": 87.46,
           "area_mean": 566.3,
           "smoothness_mean": 0.09779,
           "compactness_mean": 0.08129,
           "concavity_mean": 0.06664,
           "concave_points_mean": 0.04781,
           "symmetry_mean": 0.1885,
           "fractal_dimension_mean": 0.05766,
           "radius_se": 0.2699,
           "texture_se": 0.7886,
           "perimeter_se": 2.058,
           "area_se": 23.56,
           "smoothness_se": 0.008462,
           "compactness_se": 0.0146,
           "concavity_se": 0.02387,
           "concave_points_se": 0.01315,
           "symmetry_se": 0.0198,
           "fractal_dimension_se": 0.0023,
           "radius_worst": 15.11,
           "texture_worst": 19.26,
           "perimeter_worst": 99.7,
           "area_worst": 711.2,
           "smoothness_worst": 0.144,
           "compactness_worst": 0.1773,
           "concavity_worst": 0.239,
           "concave_points_worst": 0.1288,
           "symmetry_worst": 0.2977,
           "fractal_dimension_worst": 0.07259
         }'
```
