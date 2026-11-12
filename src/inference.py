"""Funções simples de inferência reutilizáveis em serviços compatíveis."""
import json
import joblib
import pandas as pd


def load_model(model_path="artifacts/model.joblib"):
    return joblib.load(model_path)


def predict_records(records, model):
    frame = pd.DataFrame(records)
    predictions = model.predict(frame).tolist()
    probabilities = model.predict_proba(frame).tolist() if hasattr(model, "predict_proba") else None
    return {"predictions": predictions, "probabilities": probabilities}


def predict_json(payload, model):
    data = json.loads(payload) if isinstance(payload, str) else payload
    records = data.get("instances", data) if isinstance(data, dict) else data
    return predict_records(records, model)
