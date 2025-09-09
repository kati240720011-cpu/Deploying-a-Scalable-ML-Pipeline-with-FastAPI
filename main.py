from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from ml.model import Artifacts

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "model")

app = FastAPI(title="Census Income Inference API")

class CensusRequest(BaseModel):
    # match dataset columns (snake_case mapped to hyphenated where needed)
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int | None = None
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

def _normalize_payload(data: dict) -> pd.DataFrame:
    mapping = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
    }
    d = dict(data)
    for k, v in list(d.items()):
        if k in mapping:
            d[mapping[k]] = d.pop(k)
    return pd.DataFrame([d])

@app.get("/")
def root():
    return {"message": "Welcome to the Census Income Inference API"}

@app.post("/predict")
def predict(payload: CensusRequest):
    arts = Artifacts.load(ARTIFACT_DIR)
    X = _normalize_payload(payload.dict())
    proba = float(arts.model.predict_proba(X)[0, 1]) if hasattr(arts.model, "predict_proba") else None
    pred = arts.model.predict(X)[0]
    return {"prediction": str(pred), "probability_gt50k": proba}
