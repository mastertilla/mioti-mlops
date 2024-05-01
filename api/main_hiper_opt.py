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
    "ApprovalMonth": "Dec",
    "ApprovalYear": 92,
    "Sector": "Unknown Sector",
    "TermGroup": "6-12 months",
    "NewBankState": "SD",
    "IsFranchise": "Y",
    "NewCity": "YUBA CITY",
    "NewBank": "WELLS FARGO BANK NATL ASSOC"
}
ChargeOff:0
"""

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import pdb

model = joblib.load('best_clasificador_loan.pkl')
app_loan_hiper_opt = FastAPI()

"""
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
"""
def data_prep(message: dict):
    pdb.set_trace()

    columns = ['NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'UrbanRural', 'RevLineCr', 'LowDoc',
               'ApprovalYear', 'Sector', 'TermGroup', 'NewBankState', 'IsFranchise', 'NewCity', 'NewBank']
    data = pd.DataFrame(columns=columns)
    data = data.append(message, ignore_index=True)

    return data


def loan_prediction(data: pd.DataFrame):
    pdb.set_trace()
    label = model.predict(data)[0]
    return {'label': int(label)}

@app_loan_hiper_opt.post('/concede_loan_hiper_opt__prediction/')
def predict_loan(message: dict):
    pdb.set_trace()
    data = data_prep(message)
    model_pred = loan_prediction(data)
    return model_pred


"""
from fastapi.responses import JSONResponse
@app_loan_hiper_opt.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )
"""