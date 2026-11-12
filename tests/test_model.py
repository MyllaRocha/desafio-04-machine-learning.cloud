from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_saved_model_predicts_one_record():
    model_path = ROOT / "artifacts" / "model.joblib"
    csv_path = ROOT / "data" / "Social_media_impact_on_life.csv"
    assert model_path.exists(), "Execute python src/train.py antes dos testes."
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    sample = df.drop(columns=["Overall_Impact", "Student_ID"]).iloc[[0]]
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in {"Beneficial", "Neutral", "Negative"}
