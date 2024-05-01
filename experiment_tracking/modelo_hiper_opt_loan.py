import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.feature_selection import SelectPercentile, RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_validate, cross_val_score, RepeatedStratifiedKFold,\
StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve, confusion_matrix, recall_score, precision_score
from sklearn.exceptions import ConvergenceWarning
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy.stats import chi2_contingency
from scipy import stats
import seaborn as sns

import optuna

from datetime import datetime, date

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from typing import Union
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

import mlflow
warnings.simplefilter('ignore')

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('mioti-loan-hiper_opt')


sba=pd.read_csv("sba_more_reduced.csv", sep=',')

##Número de filas y columnas que tenemos
print(sba.shape)

##Función para visualización ¿Poner en la API los gráficos? Seguridad(Token) Tipado. Funciones que chequeen los datos. Tipos de errores.


def preprocessing(sba: pd.DataFrame):
    sba.drop('ChgOffDate', axis=1, inplace=True)

    top_5_state = sba['State'].value_counts(normalize=True).sort_values(ascending=False).head().index
    sba5 = sba[sba['State'] == 'CA'].copy()
    sba5.drop('State', axis=1, inplace=True)

    sba5.dropna(subset='MIS_Status', inplace=True)
    sba5['MIS_Status'] = np.where(sba5['MIS_Status'] == 'P I F', 0, 1)

    # Nos quedamos con el mes y año de aprobación
    sba5['ApprovalMonth'] = sba5['ApprovalDate'].str.split('-').str[1]
    sba5['ApprovalYear'] = sba5['ApprovalDate'].str.split('-').str[2]
    sba5['ApprovalYear'] = sba5['ApprovalYear'].apply(pd.to_numeric)
    sba5.drop('ApprovalDate', axis=1, inplace=True)

    #Transformamos cada valor en su sector al que pertenezca con la función de abajo get_sector
    sba5['NAICS'] = sba5['NAICS'].astype('string')

    naics_to_sector = {
        '11': 'Agriculture, Forestry, Fishing and Hunting',
        '21': 'Mining, Quarrying, and Oil and Gas Extraction',
        '22': 'Utilities',
        '23': 'Construction',
        '31-33': 'Manufacturing',
        '42': 'Wholesale Trade',
        '44-45': 'Retail Trade',
        '48-49': 'Transportation and Warehousing',
        '51': 'Information',
        '52': 'Finance and Insurance',
        '53': 'Real Estate and Rental and Leasing',
        '54': 'Professional, Scientific, and Technical Services',
        '55': 'Management of Companies and Enterprises',
        '56': 'Administrative and Support and Waste Management and Remediation Services',
        '61': 'Educational Services',
        '62': 'Health Care and Social Assistance',
        '71': 'Arts, Entertainment, and Recreation',
        '72': 'Accommodation and Food Services',
        '81': 'Other Services (except Public Administration)',
        '92': 'Public Administration'
    }

    def sector(naics_code: str):
        first_two = naics_code[:2]
        if first_two in ['31', '44', '48']:
            if first_two == '31':
                return naics_to_sector['31-33']
            elif first_two == '44':
                return naics_to_sector['44-45']
            else:
                return naics_to_sector['48-49']
        else:
            return naics_to_sector.get(first_two, 'Unknown Sector')

    sba5['Sector'] = sba5['NAICS'].apply(sector)
    sba5.drop('NAICS', axis=1, inplace=True)


    #Cambiamos a categórica la vairable Terms
    sba5['TermGroup'] = np.where(sba5['Term'] <= 90, 'Below 3 months',
                                 np.where((sba5['Term'] > 90) & (sba5['Term'] <= 180), '3-6 months',
                                          np.where((sba5['Term'] > 180) & (sba5['Term'] <= 365), '6-12 months',
                                                   'More Than a Year')))
    sba5.drop('Term', axis=1, inplace=True)


    #
    sba5 = sba5[sba5['NewExist'] != 0]
    sba5 = sba5.dropna(subset='NewExist')

    #Variable que no tiene valores mas que 0
    sba5.drop('BalanceGross', axis=1, inplace=True)

    #Me da igual la cantidad a devolver, la cantidad garantizada del prestamo, cantidad desembolsada, y el ejercicio fiscal de compromiso
    sba5.drop(['ChgOffPrinGr','SBA_Appv', 'DisbursementGross', 'ApprovalFY'], axis=1, inplace=True)


    #Si tienen línea de crédito
    sba5 = sba5[sba5['RevLineCr'].isin(['N', 'Y'])]
    #Si necesitan préstamo de menos de 150k
    sba5 = sba5[sba5['LowDoc'].isin(['N', 'Y'])]

    #Eliminamos valores nulos porque ya hay datos suficientes
    sba5.dropna(subset=['Bank', 'BankState', 'City', 'LowDoc'], inplace=True)

    count_bankst = sba5['BankState'].value_counts().to_frame().reset_index().rename(columns={'count': 'BankstCount'})
    count_bankst['BankState'] = count_bankst['BankState'].astype(str)
    sba5 = sba5.merge(count_bankst, on='BankState', how='left')
    bankstate_counts = sba5['BankState'].value_counts()
    sba5['NewBankState'] = np.where(bankstate_counts[sba5['BankState']].values > 15, sba5['BankState'], 'OTHER')
    sba5.drop(['BankState'], axis=1, inplace=True)
    sba5.drop(['Name', 'LoanNr_ChkDgt', 'Zip'], axis=1, inplace=True)

    sba5['IsFranchise'] = np.where(((sba5['FranchiseCode'] == 1) | (sba5['FranchiseCode'] == 0)), 'Y', 'N')
    sba5.drop('FranchiseCode', axis=1, inplace=True)

    sba5.drop(columns=['DisbursementDate'], inplace=True)
    if 'index_x' in sba5.columns and 'index_y' in sba5.columns:
        sba5.drop(columns=['index_x', 'index_y'], inplace=True)
    sba5['City'] = sba5['City'].str.upper()

    count_city = sba5['City'].value_counts().to_frame().reset_index().rename(
        columns={'index': 'City', 'City': 'City_Count'})
    sba5 = sba5.merge(count_city, on='City')
    sba5.drop(['City', 'City_Count'], axis=1, inplace=True)

    sba5['Bank'] = sba5['Bank'].str.upper()

    count_bank = sba5['Bank'].value_counts().to_frame().reset_index().rename(
        columns={'index': 'Bank', 'Bank': 'Bank_Count'})

    sba5 = sba5.merge(count_bank, on='Bank')

    sba5['NewBank'] = np.where(sba5['Bank_Count'] > 100, sba5['Bank'], 'OTHER')

    sba5.drop(['Bank', 'Bank_Count'], axis=1, inplace=True)

    sba5['UrbanRural'] = np.where(sba5['UrbanRural'] == 1, 'Urban',
                                  np.where(sba5['UrbanRural'] == 2, 'Rural', 'Undefined'))

    sba5.drop('GrAppv', axis=1, inplace=True)
    sba5.rename(columns={'MIS_Status': 'ChargeOff'}, inplace=True)
    sba5['ChargeOff'] = sba5['ChargeOff'].astype('category')

    from sklearn.model_selection import train_test_split


    return sba5


#Ahora aqui va el Hiper_mlflow (Ver archivo)
sba=pd.read_csv("sba_more_reduced.csv", sep=',')
sba5 = preprocessing(sba)
X = sba5.drop(columns=['ChargeOff', 'ApprovalMonth'])
y = sba5['ChargeOff']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=sba5['ChargeOff'], random_state=1000)

space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400]),
    "min_samples_split": hp.choice("min_samples_split", [3,7,10,15]),
    "max_depth": hp.choice("max_depth", [1, 3, 5, 7, 8]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

#Para construir la pipeline
one_hot_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(exclude='object').columns

rfc = RandomForestClassifier(max_depth=7, min_samples_split=10, random_state=1000)
smote = SMOTE(random_state=1000)
rfc_pipe_num = Pipeline([
    ('scaler', RobustScaler()),
])
rfc_pipe_cat = Pipeline([
    ('onehot', OneHotEncoder(drop='first')),
])

rfc_transformer = ColumnTransformer([
    ('pipe_num', rfc_pipe_num, numeric_cols),
    ('pipe_cat', rfc_pipe_cat, one_hot_cols)
])

def objective(params: dict[str, Union[str, int]]):
    with mlflow.start_run():
        mlflow.set_tag('model', 'Random Forest hiper opt')
        mlflow.log_params(params)

        #mlflow.log_param('Balanceo variable objetivo', Counter(y))
        mlflow.log_params(params)

        rfc_hiper_opt = RandomForestClassifier(**params, n_jobs=-1)
        rfc_pipe_combine = Pipeline([
                ('transformer', rfc_transformer),
                ('rfe', RFE(rfc)),
                ('resampling', smote),
                ('rfc', rfc_hiper_opt)
            ])

        rfc_pipe_combine.fit(X_train_val, y_train_val)
        y_pred = rfc_pipe_combine.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', metrics.precision_score(y_test, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test, y_pred))

    return {'loss': 1 - metrics.recall_score(y_test, y_pred), 'status': STATUS_OK}

#Lo dejo en 10 trials porque no tengo tiempo. Ya lo ejecute para 50 y se los resultados que salen, pero elimine el repositorio
#Y volvi a empezar de nuevo. Esta ejecución la realizo para que se tenga constancia de ello
#Los mejores parámetros que utilizo son los que tenía antes, de la ejecución con el Dataset de 900k ejemplos y 50 pruebas
best_result = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials()
)

import pickle
rfc_hiper_opt = RandomForestClassifier(**best_result, n_jobs=-1)
rfc_pipe_combine = Pipeline([
    ('transformer', rfc_transformer),
    ('rfe', RFE(rfc)),
    ('resampling', smote),
    ('rfc', rfc_hiper_opt)
])


