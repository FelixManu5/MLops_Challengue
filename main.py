"""
Datos de entrada del modelo:
['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'gender_Other', 'ever_married_Yes',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes']

{
    'age': int,
    'hypertension': int (1/0),
    'gender': str (male/female/other),
    'ever_married_Yes': int (1/0),
    'heart_disease': int (1/0),
    'avg_glucose_level': int,
    'bmi': int,
    'work_type': str (never worked/private/self-employed/children)
    'residence_type': str (urban)
    'smoking_status': str (formerly smoked/never smoked/smokes)
}

{
    "age": 33,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 0,
    "avg_glucose_level": 70,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

{
    "age": 75,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 1,
    "avg_glucose_level": 120,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

"""
from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load('model.sav')

app = FastAPI()

def gender_encoding(message):
    gender_encoded = {'gender_Male': 0, 'gender_Other': 0}
    if message['gender'].lower() == 'male':
        gender_encoded['gender_Male'] = 1
    elif message['gender'].lower() == 'other':
        gender_encoded['gender_Other'] = 1

    del message['gender']

    return message.update(gender_encoded)

def work_type_encoding(message):
    work_type_encoded = {'work_type_Never_worked': 0, 'work_type_Private': 0,
                         'work_type_Self-employed': 0, 'work_type_children': 0}

    if message['work_type'].lower() == 'never worked':
        work_type_encoded['work_type_Never_worked'] = 1
    elif message['work_type'].lower() == 'private':
        work_type_encoded['work_type_Private'] = 1
    elif message['work_type'].lower() == 'self-employed':
        work_type_encoded['work_type_Self-employed'] = 1
    elif message['work_type'].lower() == 'children':
        work_type_encoded['work_type_children'] = 1

    del message['work_type']

    return message.update(work_type_encoded)

def residence_encoding(message):
    residence_encoded = {'Residence_type_Urban': 0}
    if message['residence_type'] == 'urban':
        residence_encoded['Residence_type_Urban'] = 1

    del message['residence_type']

    return message.update(residence_encoded)

def smoking_encoding(message):
    smoking_encoded = {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 0,
                       'smoking_status_smokes': 0}
    if message['smoking_status'] == 'formerly smoked':
        smoking_encoded['smoking_status_formerly smoked'] = 1
    elif message['smoking_status'] == 'never smoked':
        smoking_encoded['smoking_status_never smoked'] = 1
    elif message['smoking_status'] == 'smokes':
        smoking_encoded['smoking_status_smokes'] = 1

    del message['smoking_status']

    return message.update(smoking_encoded)

def data_prep(message):
    gender_encoding(message)
    work_type_encoding(message)
    residence_encoding(message)
    smoking_encoding(message)

    return pd.DataFrame(message, index=[0])


def heart_prediction(message: dict):
    # Data Prep
    data = data_prep(message)
    label = model.predict(data)[0]
    return {'label': int(label)}



@app.get('/')
def main():
    return {'message': 'Hola'}

@app.post('/heart-attack-prediction/')
def predict_heart_attack(message: dict):
    model_pred = heart_prediction(message)
    # return {'prediction': model_pred}
    return model_pred