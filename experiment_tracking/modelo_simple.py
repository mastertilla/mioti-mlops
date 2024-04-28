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
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


df = pd.read_csv('healthcare-dataset-stroke-data.csv', sep=',')
print("##### Data Preprocessing #####\n")
print(f'Numero de datos que tenemos: {len(df)}\n')

def cat_to_num_variables(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            encoded_labels = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, encoded_labels], axis=1)

    return df

def data_preprocessing(data):
    # Convertimos todos los datos nulos a la media de los valores de esa columna
    data = data.fillna(data.mean())
    # Quitamos la columna Id porque no nos aporta
    data = data.drop(columns=['id'])
    # Convertimos las variables categoricas a numericas con get_dummies
    data = cat_to_num_variables(data)

    return data

df = data_preprocessing(df)
print(df.head(5).to_markdown())

print("\n##### Dataset Balancing #####\n")
# Primero hacemos un split de nuestros datos entre variables independientes y variables objetivo
X = df.drop(columns=['stroke'])
y = df['stroke']
print(f'Numero de casos de no infarto vs infarto: {Counter(y)}')

def dataset_oversampling(X, y):
    # Definimos la estrategia de oversampling
    over = RandomOverSampler(sampling_strategy=0.2)
    # Adaptamos a nuestro dataset
    X_over, y_over = over.fit_resample(X, y)
    # summarize class distribution
    print(f'Numero de casos de no infarto vs infarto despues de oversampling: {Counter(y_over)}')

    return X_over, y_over

def dataset_undersampling(X, y):
    # Definimos la estrategia de oversampling
    under = RandomUnderSampler(sampling_strategy=0.5)
    # Adaptamos a nuestro dataset
    X_over_under, y_over_under = under.fit_resample(X, y)
    # summarize class distribution
    print(f'Numero de casos de no infarto vs infarto despues de oversampling y undersampling: {Counter(y_over_under)}')

    return X_over_under, y_over_under

X, y = dataset_oversampling(X, y)
X, y = dataset_undersampling(X, y)

print("\n##### Model Training #####\n")

# Hacemos un split de nuestros datos
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X, y, test_size=.1)

# Creamos un modelo de Random Forest, y lo entrenamos
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train_balanced, y_train_balanced)

y_pred = clf.predict(X_test_balanced)

print(f'Accuracy of the model: {metrics.accuracy_score(y_test_balanced, y_pred)}, '
      f'precision of the model: {metrics.precision_score(y_test_balanced, y_pred)}, '
      f'recall: {metrics.recall_score(y_test_balanced, y_pred)}')