"""
Datos de entrada del modelo:

{'NoEmp': int, #Condicion de que negativos no valen
 'NewExist': float, #Si pone int pasar a float
 'CreateJob': int,
 'RetainedJob': int,
 'UrbanRural': 'Undefined/Urban/Rural, #Necesitamos hacer lower y poner en mayuscula la primera letra.
 'RevLineCr': Y/N,
 'LowDoc': Y/N,
 'ApprovalYear': 70/71/72.../0/1/2/3.../11/12/13,
 'Sector': Unknown Sector/Professional, Scientific, and Technical Services/Retail Trade/Wholesale Trade/Other Services (except Public Administration)
 /Construction/Health Care and Social Assistance/Accommodation and Food Services/Administrative and Support and Waste Management and Remediation Services
 /Manufacturing/Transportation and Warehousing/Real Estate and Rental and Leasing/Information/Finance and Insurance/Arts, Entertainment, and Recreation
 /Educational Services/Agriculture, Forestry, Fishing and Hunting/Mining, Quarrying, and Oil and Gas Extraction/Utilities
 /Public Administration/Management of Companies and Enterprises,
 'TermGroup': 3-6 months/6-12 months/More Than a Year,
 'NewBankState': CA/NC/SD/OH/IL/VA/TX/NY/OR/DC/CO/MO/UT/FL/NV/MD/OTHER/WA/MA/AL/SC/CT/AZ/TN/MN/HI,
 'IsFranchise': Y/N,
 'NewCity': OTHER/LOS ANGELES/SAN DIEGO/SAN FRANCISCO/SAN JOSE/...../SAN JUAN CAPISTRANO/WOODLAND/ARROYO GRANDE/ALAMEDA/RANCHO SANTA MARGARITA  ¿Que hago aquí?
 'NewBank': 'RABOBANK, NATIONAL ASSOCIATION'}

{
    "NoEmp": 3,
    "NewExist": 2.0,
    "CreateJob": 0,
    "RetainedJob": 0,
    "UrbanRural": "Undefined",
    "RevLineCr": "N",
    "LowDoc": "Y",
    "ApprovalYear": 97,
    "Sector": "Unknown Sector",
    "TermGroup": "3-6 months",
    "NewBankState": "CA",
    "IsFranchise": "Y",
    "NewCity": "SANTA MARIA",
    "NewBank": "RABOBANK, NATIONAL ASSOCIATION"
}


{
    "NoEmp": 3,
    "NewExist": 1.0,
    "CreateJob": 1,
    "RetainedJob": 4,
    "UrbanRural": "Urban",
    "RevLineCr": "Y",
    "LowDoc": "N",
    "ChargeOff": 1,
    "ApprovalMonth": "Apr",
    "ApprovalYear": 6,
    "Sector": "Professional, Scientific, and Technical Services",
    "TermGroup": "Below 3 months",
    "NewBankState": "SD",
    "IsFranchise": "Y",
    "NewCity": "CHICO",
    "NewBank": "WELLS FARGO BANK NATL ASSOC"
}

{
    "NoEmp": 11,
    "NewExist": 1.0,
    "CreateJob": 0,
    "RetainedJob": 0,
    "UrbanRural": "Undefined",
    "RevLineCr": "N",
    "LowDoc": "N",
    "ChargeOff": 0,
    "ApprovalMonth": "Dec",
    "ApprovalYear": 92,
    "Sector": "Unknown Sector",
    "TermGroup": "6-12 months",
    "NewBankState": "SD",
    "IsFranchise": "Y",
    "NewCity": "YUBA CITY",
    "NewBank": "WELLS FARGO BANK NATL ASSOC"
}

"""


from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

model = joblib.load('modelo_loan.pkl')
app_loan = FastAPI()

class LoanData(BaseModel):
    NoEmp: int
    NewExist: float
    CreateJob: int
    RetainedJob: int
    UrbanRural: str
    RevLineCr: str
    LowDoc: str
    ApprovalYear: int
    Sector: str
    TermGroup: str
    NewBankState: str
    IsFranchise: str
    NewCity: str
    NewBank: str

def data_prep(message:dict):
    return pd.DataFrame(message, index=[0])

def loan_prediction(data: pd.DataFrame):
    label = model.predict(data)[0]
    return {'label': int(label)}


#Api en http://127.0.0.1:8000/docs#/
@app_loan.post('/concede_loan_prediction/')
def predict_loan(message: dict):
    data = data_prep(message)
    model_pred = loan_prediction(data)
    # return {'prediction': model_pred}
    return model_pred

@app_loan.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )