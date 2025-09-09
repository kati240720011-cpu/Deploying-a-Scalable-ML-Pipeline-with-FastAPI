"""
Local client to exercise the API.
Run: uvicorn main:app --reload   # then python local_api.py
"""
import requests

BASE = "http://127.0.0.1:8000"
print("GET /")
r = requests.get(BASE + "/")
print(r.status_code, r.json())

sample = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}
print("POST /predict")
r = requests.post(BASE + "/predict", json=sample)
print(r.status_code, r.json())
