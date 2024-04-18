# Importing necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import mlflow
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('simple-exp')

def data_preprocessing(data):
    # Convertimos todos los datos nulos a la media de los valores de esa columna
    data = data.fillna(data.mean())
    # Quitamos la columna Id porque no nos aporta
    data = data.drop(columns=['id'])

    return data

def map_values(input_list):
    """
    Mapea los valores 'M' a 1 y 'B' a 0 en input_list.

    Args:
    - input_list (list): La lista de valores que se van a mapear.

    Returns:
    - list: Una lista de los valores mapeados.
    """
    mapping = {'M': 1, 'B': 0}
    return [mapping[value] for value in input_list if value in mapping]

df = pd.read_csv("breast-cancer.csv")
df.sample()

print("##### Data Preprocessing #####\n")
print(f'Numero de datos que tenemos: {len(df)}\n')

print("\n##### Dataset Balancing #####\n")
# Dividir los datos en características (X) y etiquetas (y)
X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"]

print(f'Numero de casos de no infarto vs infarto: {Counter(y)}')

y = map_values(y)

df = data_preprocessing(df)
df.head(5)

print("\n##### Model Training #####\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un modelo de Random Forest, y lo entrenamos
with mlflow.start_run():
    mlflow.set_tag('Author', 'Miguel')
    mlflow.set_tag('Model', 'Random Forest')

    # Log param information
    # Logeamos sobre los datos
    mlflow.log_param('Balanceo variable objetivo', Counter(y))
    n_estimators = 100
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=3,
                                 criterion='gini', random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mlflow.log_metric('accuracy', metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', metrics.precision_score(y_test, y_pred))
    mlflow.log_metric('recall', metrics.recall_score(y_test, y_pred))



    with open('models/clasificador.pkl', 'wb') as f_out:
        pickle.dump(clf, f_out)

    mlflow.log_artifact(local_path='models/clasificador.pkl', artifact_path='models_pickle')
