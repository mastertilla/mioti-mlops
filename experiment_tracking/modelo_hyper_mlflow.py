# %%
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import mlflow
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# %%
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('hyperopt-exp')

# %%
def cat_to_num_variables(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            encoded_labels = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, encoded_labels], axis=1)

    return df

def data_preprocessing(data):
    # Convertimos las variables categoricas a numericas con get_dummies
    data = cat_to_num_variables(data)
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


# %%
df = pd.read_csv("breast-cancer.csv")
df.sample()

# %%
print("##### Data Preprocessing #####\n")
print(f'Numero de datos que tenemos: {len(df)}\n')

# %%
print("\n##### Dataset Balancing #####\n")
# Dividir los datos en características (X) y etiquetas (y)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
print(f'Numero de casos de no infarto vs infarto: {Counter(y)}')

# %%
y = map_values(y)

# %%
df = data_preprocessing(df)
df.head(5)

# %%
print("\n##### Model Training #####\n")

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.choice("max_depth", [1, 2, 3, 5, 8]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag('model', 'Random Forest')
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', metrics.precision_score(y_test, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test, y_pred))

    return {'loss': 1 - metrics.recall_score(y_test, y_pred), 'status': STATUS_OK}

best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )


