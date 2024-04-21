Para utilizar la API, los usuarios deben enviar datos en el siguiente formato JSON.
La API recibe datos específicos del cliente y devuelve una predicción de churn:

0: Indica que se espera que el cliente no deje la empresa.
1: Indica que se espera que el cliente deje la empresa.

Datos originales (de entrada):
CreditScore: int64 (Entero)
Geography: object (Cadena de texto; opciones: "France", "Germany", "Spain")
Gender: object (Cadena de texto; opciones: "Male", "Female")
Age: float64 (Decimal)
Tenure: category (Categórico; opciones numéricas de 0 a 10)
Balance: float64 (Decimal)
NumOfProducts: category (Categórico; opciones numéricas de 1 a 4)
HasCrCard: category (Categórico; opciones binarias 0.0 o 1.0)
IsActiveMember: category (Categórico; opciones binarias 0.0 o 1.0)
EstimatedSalary: float64 (Decimal)

Datos transformados (después del preprocesamiento):
CreditScore: float64
Age: float64
Balance: float64
EstimatedSalary: float64
Geography_[Country]: float64 para cada país (France, Germany, Spain)
Gender_[Gender]: float64 para cada género (Male, Female)
Tenure_[0-10]: float64 para cada año de tenure
NumOfProducts_[1-4]: float64 para cada número de productos
HasCrCard_[0.0, 1.0]: float64 para cada estado de tarjeta de crédito
IsActiveMember_[0.0, 1.0]: float64 para cada estado de miembro activo
Active_with_cr_[0.0, 1.0]: float64 para cada combinación de miembro activo y tarjeta de crédito
Products_per_year_[0-10]: float64 para cada relación productos/año
Zero_Balance_[0, 1]: float64 para indicar si el balance es cero o no
Old_[0, 1]: float64 para indicar si la edad es mayor a 50 años o no

Ejemplo 1: (label 0)
{"CreditScore": 702, "Geography": "France", "Gender": "Male", "Age": 37.0, 
"Tenure": 10, "Balance": 150525.8, "NumOfProducts": 1, "HasCrCard": 1.0, 
"IsActiveMember": 1.0, "EstimatedSalary": 94728.49}

Ejemplo 2: (label 1)
{"CreditScore": 601, "Geography": "France", "Gender": "Female", "Age": 43.0, 
"Tenure": 8, "Balance": 0.0, "NumOfProducts": 3, "HasCrCard": 0.0, 
"IsActiveMember": 1.0, "EstimatedSalary": 110916.15}

