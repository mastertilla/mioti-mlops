import numpy as np
import pandas as pd
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv('health_dataset.csv')
print(f'Numero de datos que tenemos: {len(df)}\n')

print("Preprocessing...\n")

df['day'] = pd.to_datetime(df['day'])
df['timestamp'] = df['day'] + pd.to_timedelta(df['hour'], unit='h')
df.set_index('timestamp', inplace=True)

X = np.array(df[['sex','age','weight','hour','bpm']])
df.status = df.status.map({'OK': 1, 'BAD': 0})
y = np.array(df.status)

print("Model Training...\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

xgboost = XGBClassifier(random_state=42)
xgboost.fit(X_train, y_train)

y_predicted = xgboost.predict(X_test)
y_predicted_prob = xgboost.predict_proba(X_test)

print("precision_score: {}".format(precision_score(y_test,y_predicted)))
print("recall_score: {}".format(recall_score(y_test,y_predicted)))
print("F1: {}".format(f1_score(y_test,y_predicted)))
print("AUC: {}".format(roc_auc_score(y_test,y_predicted_prob[:,1])))

joblib.dump(xgboost, '../api/model.sav')