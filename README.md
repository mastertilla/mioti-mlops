# MLOps @ Mioti

## API
La API se lanza con

```bash
uvicorn main:app --reload
```
Una vez levantada, se puede acceder en : [http://127.0.0.1:8000](http://127.0.0.1:8000) y probar valores en : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Entrenamiento del modelo
El modelo utilizado fue Random forest entrenado con el dataset de diabetes de sklearn. Recibe las variables    


1. `age`: float (Edad del paciente)
2. `sex`: str (Sexo del paciente, "male" o "female")
3. `bmi`: float (Índice de masa corporal)
4. `bp`: float (Presión arterial)
5. `cholesterol`: float (Nivel de colesterol)
6. `fbs`: float (Nivel de azúcar en ayunas)
7. `restecg`: float (Resultados del electrocardiograma en reposo)
8. `thalach`: float (Frecuencia cardíaca máxima alcanzada)
9. `exang`: float (Angina inducida por el ejercicio, 1 = sí, 0 = no)
10. `oldpeak`: float (Depresión del ST inducida por el ejercicio en relación con el reposo)


## Ejemplo de uso
Para hacer una predicción, puedes enviar una solicitud POST a la ruta `/diabetes-prediction/` con el siguiente formato:
```json
{
    "age": 50,
    "sex": "male",
    "bmi": 25.0,
    "bp": 80.0,
    "cholesterol": 200.0,
    "fbs": 100.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 1.5
}

Se predice 1 cuando hay diabetes y 0 cuando no hay


