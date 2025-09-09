import pandas as pd
from ml.model import train_model, compute_model_metrics, performance_on_categorical_slice

def _dummy():
    df = pd.DataFrame({
        "workclass": ["A","B","A","B","A","B"],
        "education": ["HS","HS","BS","BS","MS","MS"],
        "marital-status": ["Never-married"]*6,
        "occupation": ["Tech"]*6,
        "relationship": ["Self"]*6,
        "race": ["Other"]*6,
        "sex": ["Female","Male","Female","Male","Female","Male"],
        "native-country": ["US"]*6,
        "age": [25,35,45,28,30,40],
        "fnlgt": [1,2,3,4,5,6],
        "education-num": [9,9,13,13,14,14],
        "capital-gain": [0,0,0,0,0,0],
        "capital-loss": [0,0,0,0,0,0],
        "hours-per-week": [40,40,40,40,40,40],
        "salary": [0,1,1,0,1,0],
    })
    X = df.drop(columns=["salary"])
    y = df["salary"]
    return X, y, df

def test_train_and_predict_len():
    X, y, _ = _dummy()
    model = train_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)

def test_metrics_bounds():
    X, y, _ = _dummy()
    model = train_model(X, y)
    preds = model.predict(X)
    p, r, f1 = compute_model_metrics(y, preds)
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0 and 0.0 <= f1 <= 1.0

def test_slice_metrics_cover_values():
    X, y, df = _dummy()
    model = train_model(X, y)
    df2 = X.copy(); df2["salary"] = y
    res = performance_on_categorical_slice(model, df2, feature="education")
    assert set(res.keys()) == {"HS","BS","MS"}
