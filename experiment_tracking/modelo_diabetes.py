import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Definir la ruta donde se guardará el modelo
model_path = os.path.join('..', 'api', 'model.sav')

# Cargar el dataset de diabetes
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# Convertimos el problema a clasificación binaria: nivel alto (1) o bajo (0) de progresión de diabetes
y = (diabetes.target > diabetes.target.mean()).astype(int)

# Renombrar las columnas para que sean más comprensibles
X.rename(columns={
    'age': 'age',
    'sex': 'sex',
    'bmi': 'bmi',
    'bp': 'bp',
    's1': 'cholesterol',
    's2': 'fbs',
    's3': 'restecg',
    's4': 'thalach',
    's5': 'exang',
    's6': 'oldpeak'
}, inplace=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el modelo en un archivo .sav
joblib.dump(model, model_path)
