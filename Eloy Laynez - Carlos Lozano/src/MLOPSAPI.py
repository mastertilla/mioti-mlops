# %%
''' Datos de entrada del usuario

<class 'pandas.core.frame.DataFrame'>
Index: 9998 entries, 0 to 10001
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype   
---  ------           --------------  -----   
 0   CreditScore      9998 non-null   int64   
 1   Geography        9998 non-null   object  
 2   Gender           9998 non-null   object  
 3   Age              9998 non-null   float64 
 4   Tenure           9998 non-null   category
 5   Balance          9998 non-null   float64 
 6   NumOfProducts    9998 non-null   category
 7   HasCrCard        9998 non-null   category
 8   IsActiveMember   9998 non-null   category
 9   EstimatedSalary  9998 non-null   float64 
dtypes: category(4), float64(3), int64(1), object(2)
memory usage: 586.6+ KB

Ojo debemos hacer un pipeline, los datos tienen que ser transformados como el se entreno el modelo. son los siguientes:


<class 'pandas.core.frame.DataFrame'>
Index: 7877 entries, 1768 to 7384
Data columns (total 45 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   CreditScore           7877 non-null   float64
 1   Age                   7877 non-null   float64
 2   Balance               7877 non-null   float64
 3   EstimatedSalary       7877 non-null   float64
 4   Geography_France      7877 non-null   float64
 5   Geography_Germany     7877 non-null   float64
 6   Geography_Spain       7877 non-null   float64
 7   Gender_Female         7877 non-null   float64
 8   Gender_Male           7877 non-null   float64
 9   Tenure_0              7877 non-null   float64
 10  Tenure_1              7877 non-null   float64
 11  Tenure_2              7877 non-null   float64
 12  Tenure_3              7877 non-null   float64
 13  Tenure_4              7877 non-null   float64
 14  Tenure_5              7877 non-null   float64
 15  Tenure_6              7877 non-null   float64
 16  Tenure_7              7877 non-null   float64
 17  Tenure_8              7877 non-null   float64
 18  Tenure_9              7877 non-null   float64
 19  Tenure_10             7877 non-null   float64
 20  NumOfProducts_1       7877 non-null   float64
 21  NumOfProducts_2       7877 non-null   float64
 22  NumOfProducts_3       7877 non-null   float64
 23  NumOfProducts_4       7877 non-null   float64
 24  HasCrCard_0.0         7877 non-null   float64
 25  HasCrCard_1.0         7877 non-null   float64
 26  IsActiveMember_0.0    7877 non-null   float64
 27  IsActiveMember_1.0    7877 non-null   float64
 28  Active_with_cr_0.0    7877 non-null   float64
 29  Active_with_cr_1.0    7877 non-null   float64
 30  Products_per_year_0   7877 non-null   float64
 31  Products_per_year_1   7877 non-null   float64
 32  Products_per_year_2   7877 non-null   float64
 33  Products_per_year_3   7877 non-null   float64
 34  Products_per_year_4   7877 non-null   float64
 35  Products_per_year_5   7877 non-null   float64
 36  Products_per_year_6   7877 non-null   float64
 37  Products_per_year_7   7877 non-null   float64
 38  Products_per_year_8   7877 non-null   float64
 39  Products_per_year_9   7877 non-null   float64
 40  Products_per_year_10  7877 non-null   float64
 41  Zero_Balance_0        7877 non-null   float64
 42  Zero_Balance_1        7877 non-null   float64
 43  Old_0                 7877 non-null   float64
 44  Old_1                 7877 non-null   float64
dtypes: float64(45)
memory usage: 2.8 MB


'''

# %%
''' probar la API
{
'CreditScore'            : 772,
'Geography'              : 'Germany',
'Gender'                    : 'Male',
'Age'                       : 42.0,	
'Tenure'                       : 3,
'Balance'               : 75075.31,
'NumOfProducts'                : 2,	
'HasCrCard'                  : 1.0,	
'IsActiveMember'             : 0.0,	
'EstimatedSalary'       : 92888.52}
  
  exited : 1
  	
   
   	
'''

# %%
import pandas as pd
import numpy as np
from fastapi import FastAPI
import joblib
import xgboost as xgb





# %%
model = joblib.load(r'C:\Users\carlo\OneDrive\Escritorio\MIOTI\MLOPS\src\xgb_mlops.pkl')
scaler = joblib.load(r'C:\Users\carlo\OneDrive\Escritorio\MIOTI\MLOPS\src\StandarScaler_mlops.pkl')
#encoder = joblib.load('onehot_mlops.pkl') # depencias distintas, no puedo crear env con esta depencia, hacerlo manual
app=FastAPI()

# %% [markdown]
# Empezamos con las funciones para de preprocesado.

# %%
#Datos de entrada


# %%
def dicc_a_df(message:dict)->pd.DataFrame:
#Definir si lo que entra será un dicc

    return pd.DataFrame(message,index=[0])

#HasCrCard'  e IsActiveMember
def hasCRCard_IsActiveMember(X:pd.DataFrame):
    X['Active_with_cr']=X['HasCrCard']*X['IsActiveMember']
    X['Active_with_cr']=X['Active_with_cr'].astype('category')
    return None
#Tenure y NumOfProducts
def tenure_NumofProduct(X:pd.DataFrame):
    X['Products_per_year']=round(X['Tenure']/X['NumOfProducts'])
    X['Products_per_year']=X['Products_per_year'].astype('int').astype('category')
    return None
#Balance 
def balance(X:pd.DataFrame):
    X['Zero_Balance']=X['Balance'].apply(lambda x: 1 if x==0 else 0)
    X['Zero_Balance']=X['Zero_Balance'].astype('category')
    return None
#Age
def age(X:pd.DataFrame):
    X['Old']=X['Age'].apply(lambda x: 1 if x>50 else 0)
    X['Old']=X['Old'].astype('category')
    
    return None




# %%
#escalamos los datos numericos con el modelo entrenado
def escalado_num(X:pd.DataFrame,prueba=False):
    numerical_cols=['CreditScore','Age','Balance','EstimatedSalary']
    X[numerical_cols]=scaler.transform(X[numerical_cols])
    
    if prueba == True:
        
        return X
    else:
        return None

#one hot los datos categoricos con el modelo entrenado

import pandas as pd

def manual_one_hot(df):
    required_columns = [
        'CreditScore', 'Age', 'Balance', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain',
        'Gender_Female', 'Gender_Male',
        'Tenure_0', 'Tenure_1', 'Tenure_2', 'Tenure_3', 'Tenure_4', 'Tenure_5', 'Tenure_6', 'Tenure_7', 'Tenure_8', 'Tenure_9', 'Tenure_10',
        'NumOfProducts_1', 'NumOfProducts_2', 'NumOfProducts_3', 'NumOfProducts_4',
        'HasCrCard_0.0', 'HasCrCard_1.0',
        'IsActiveMember_0.0', 'IsActiveMember_1.0',
        'Active_with_cr_0.0', 'Active_with_cr_1.0',
        'Products_per_year_0', 'Products_per_year_1', 'Products_per_year_2', 'Products_per_year_3', 'Products_per_year_4',
        'Products_per_year_5', 'Products_per_year_6', 'Products_per_year_7', 'Products_per_year_8', 'Products_per_year_9', 'Products_per_year_10',
        'Zero_Balance_0', 'Zero_Balance_1',
        'Old_0', 'Old_1'
    ]

    # Crear un nuevo DataFrame vacío con todas las columnas requeridas
    df_final = pd.DataFrame(index=df.index, columns=required_columns)
    df_final = df_final.fillna(0)  # Inicializar todas las columnas con 0s

    # Asignar valores numéricos directamente
    numeric_columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    df_final[numeric_columns] = df[numeric_columns]

    # Columnas categóricas y su mapeo a los nombres en required_columns
    mapping = {
        'Geography': 'Geography', 'Gender': 'Gender', 'Tenure': 'Tenure',
        'NumOfProducts': 'NumOfProducts', 'HasCrCard': 'HasCrCard',
        'IsActiveMember': 'IsActiveMember', 'Active_with_cr': 'Active_with_cr',
        'Products_per_year': 'Products_per_year', 'Zero_Balance': 'Zero_Balance', 'Old': 'Old'
    }

    # Aplicar la codificación one-hot manualmente
    for column, prefix in mapping.items():
        for category in df[column].dropna().unique():
            col_name = f"{prefix}_{category}"
            if col_name in df_final.columns:
                df_final[col_name] = (df[column] == category).astype(float)

    columns_to_drop = [col for col in mapping.keys() if col in df_final.columns]
    df_final.drop(columns=columns_to_drop, inplace=True)
    df_final=df_final.astype(float)
   # df_final = df_final.applymap(lambda x: x.item() if isinstance(x, np.generic) else x)
    return df_final






# %%
def prepro(message):
    df=dicc_a_df(message)
    hasCRCard_IsActiveMember(df)
    tenure_NumofProduct(df)
    balance(df)
    age(df)
   


    escalado_num(df)
        
    df_final=manual_one_hot(df)
    #df_final = df_final.applymap(lambda x: x.item() if isinstance(x, np.generic) else x)
    return df_final
    

# %%
def preditc(message):

            df=prepro(message)
            val_pred=model.predict(df)[0] 
            val_pred_native = int(val_pred)
            return val_pred_native


@app.get('/')
def main():
    return {'message': 'Hola'}

@app.post('/prueba_mlops/')
def churn_ratio(message: dict):
    val_pred=preditc(message) 
    return {"label":val_pred}









