from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta 
from jose import JWTError, jwt
from passlib.context import CryptContext
import joblib
import pandas as pd

# Configuración de constantes
SECRET_KEY = "33262e4fcd1619f198b177fd2f9f00050298c5c744a36d721f4c1640f781ba6b"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30    

# Base de datos ficticia
fake_db = {}

# Modelo de token para respuesta
class Token(BaseModel):
    access_token: str
    token_type: str

# Modelo de datos del token
class TokenData(BaseModel):
    username: str | None = None 

# Modelo de usuario
class User(BaseModel):
    username: str
    email: str | None = None 
    full_name: str | None = None 
    disabled: bool | None = None

# Modelo de usuario en la base de datos
class UserInDB(User):
    hashed_password: str

# Contexto de cifrado para la contraseña
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Esquema de autenticación OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# Verificar que la contraseña en texto plano coincide con la contraseña cifrada
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Generar una contraseña cifrada
def get_password_hash(password):
    return pwd_context.hash(password)

# Obtener usuario de la base de datos
def get_user(db, username: str):
    if username in db:
        user_data = db[username]
        return UserInDB(**user_data)

# Autenticar usuario
def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    
    return user

# Crear token de acceso
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Obtener usuario actual
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
                                         headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception
        token_data = TokenData(username=username)

    except JWTError:
        raise credential_exception

    user = get_user(fake_db, username=token_data.username)
    if user is None:
        raise credential_exception
    
    return user

# Obtener usuario activo actual
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user

# Endpoint para obtener token de acceso
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint para obtener un token de acceso.

    Permite autenticar un usuario y generar un token de acceso.

    Args:
        form_data (OAuth2PasswordRequestForm): Datos del formulario de inicio de sesión.

    Returns:
        dict: Token de acceso.
    """
    user = authenticate_user(fake_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Has puesto mal la contraseña o el usuario...",
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint para obtener información del usuario actual
@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User=Depends(get_current_active_user)):
    """
    Endpoint para obtener información del usuario actual.

    Requiere un token de acceso válido y activo.

    Args:
        current_user (User): Usuario actual.

    Returns:
        dict: Información del usuario.
    """
    return current_user

# Endpoint para registrar un nuevo usuario
@app.post("/register", response_model=User)
async def register_user(username: str, full_name: str, email: str, password: str):
    """
    Endpoint para registrar un nuevo usuario.

    Permite crear un nuevo usuario en la base de datos ficticia.

    Args:
        username (str): Nombre de usuario.
        full_name (str): Nombre completo.
        email (str): Correo electrónico.
        password (str): Contraseña.

    Returns:
        dict: Datos del usuario registrado.
    """
    if username in fake_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(password)
    
    user_data = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False
    }
    fake_db[username] = user_data
    
    return user_data

model_penguin = joblib.load("log_regression_penguins.sav")
min_max_scaler = joblib.load("min_max_scaler.sav")


def island_encoding(message):
    island_encoded = {"island_Torgersen": 0, "island_Dream": 0}
    if message["island"].lower() == "torgersen":
        island_encoded["island_Torgersen"] = 1
    elif message["island"].lower() == "dream":
        island_encoded["island_Dream"] = 1
    del message["island"]
    return message.update(island_encoded)

def sex_encoding(message):
    sex_encoded = {"sex_Male": 0}
    if message["sex"].lower() == "male":
        sex_encoded["sex_Male"] = 1
    del message["sex"]
    return message.update(sex_encoded)

def data_prep(message):
    island_encoding(message)
    sex_encoding(message)
    instancia = pd.DataFrame(message, index=[0])[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g','island_Dream', 'island_Torgersen', 'sex_Male']]
    instancia = min_max_scaler.transform(instancia)
    return instancia 

def specie_prediction(message: dict, username: str):
    data = data_prep(message)
    label = model_penguin.predict(data)[0]
    return f"Hola {username}, el pigüino es de la especie {label}"
    
@app.post("/species_prediction/")
async def predict_species(isla: str, sexo: str, longitud_pico: float,
                          profundidad_pico: float, longitud_aleta:float,
                          peso:float, current_user: User=Depends(get_current_active_user)):
    message = {
        "bill_length_mm": longitud_pico,
        "bill_depth_mm": profundidad_pico,
        "flipper_length_mm": longitud_aleta,
        "body_mass_g": peso,
        "island": isla,
        "sex": sexo
    }
    model_pred = specie_prediction(message, current_user.username)
    return model_pred


import streamlit as st
import requests

# URL de la API
BASE_URL = "http://localhost:8000"  # Cambia esto por la dirección donde esté corriendo tu servidor FastAPI

# Función para registrar un nuevo usuario
def registration(username, full_name, email, password):
    url = f"{BASE_URL}/register"
    params = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "password": password
    }
    response = requests.post(url, params=params)
    return response

# Función para iniciar sesión
def login(username, password):
    url = f"{BASE_URL}/token"
    data = {"username": username, "password": password}
    response = requests.post(url, data=data)
    return response


# Función para hacer la predicción de la especie del pingüino
def predict_species(sexo, longitud_pico, longitud_aleta, profundidad_pico, peso, isla, token):
    url = f"{BASE_URL}/species_prediction/"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "sexo": sexo,  # Asegúrate de enviar el sexo como cadena de caracteres
        "longitud_pico": longitud_pico,
        "profundidad_pico": profundidad_pico,
        "longitud_aleta": longitud_aleta,
        "peso": peso,
        "isla": isla
    }
    response = requests.post(url, headers=headers, params=data)  # Usar json en lugar de params
    return response


# Página para registrar un nuevo usuario
def register_page():
    st.title("Registro de Usuario")
    username = st.text_input("Nombre de Usuario")
    full_name = st.text_input("Nombre Completo")
    email = st.text_input("Correo Electrónico")
    password = st.text_input("Contraseña", type="password")
    if st.button("Registrarse"):
        response = registration(username, full_name, email, password)
        if response.status_code == 200:
            st.success("Usuario registrado exitosamente")
        else:
            st.error(f"Error al registrar el usuario: {response.content.decode('utf-8')}")

# Página para iniciar sesión
def login_page():
    st.title("Inicio de Sesión")
    username = st.text_input("Nombre de Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar Sesión"):
        response = login(username, password)
        if response.status_code == 200:
            token = response.json().get("access_token")
            st.success("Inicio de sesión exitoso")
            st.session_state.token = token  # Almacenar el token en la sesión
            return True
        else:
            st.error("Inicio de sesión fallido")
            return False

# Página para hacer la predicción de la especie del pingüino
def prediction_page():
    st.title("Predicción de Especie de Pingüino")
    sexo = st.selectbox("Sexo del Pingüino", ["Male", "Female"])
    isla = st.selectbox("Isla del Pingünino", ["Torgersen", "Dream", "Biscoe"])
    longitud_pico = st.number_input("Longitud del Pico (mm)")
    longitud_aleta = st.number_input("Longitud de la Aleta (mm)")
    profundidad_pico = st.number_input("Profunidad del pico (mm)")
    peso = st.number_input("Peso del pingüno (g)")

    token = st.session_state.get("token")  # Obtener el token de la sesión
    if token:
        if st.button("Predecir"):
            response = predict_species(sexo, longitud_pico, longitud_aleta, profundidad_pico, peso, isla, token)
            if response.status_code == 200:
                st.success(response.text)
            else:
                st.error(f"Error al hacer la predicción: {response.content.decode('utf-8')}")
    else:
        st.error("Debes iniciar sesión primero")

# Función principal
def main():
    st.sidebar.title("Menú")
    if "token" not in st.session_state:
        selection = st.sidebar.radio("Ir a:", ["Registro", "Inicio de Sesión"])
        if selection == "Registro":
            register_page()
        elif selection == "Inicio de Sesión":
            if login_page():
                prediction_page()
    else:
        selection = st.sidebar.radio("Ir a:", ["Registro", "Inicio de Sesión"])
        if selection == "Registro":
            register_page()
        elif selection == "Inicio de Sesión":
            login_page()
            prediction_page()

if __name__ == "__main__":
    main()