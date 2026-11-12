"""Treina e avalia o modelo de classificação do Desafio 04."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "Social_media_impact_on_life.csv"
OUT = ROOT / "artifacts"
TARGET = "Overall_Impact"
SEED = 42
LABELS = ["Beneficial", "Neutral", "Negative"]


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"CSV não encontrado em {CSV}. Coloque a base em data/Social_media_impact_on_life.csv.")
    df = pd.read_csv(CSV)
    required = {TARGET, "Student_ID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if TARGET not in df or df[TARGET].isna().any():
        raise ValueError("A coluna-alvo está ausente ou contém valores nulos.")
    X = df.drop(columns=[TARGET, "Student_ID"], errors="ignore")
    y = df[TARGET].astype(str)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, random_state=SEED))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]}, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "best_cv_f1_macro": float(search.best_score_), "best_params": search.best_params_,
        "labels": LABELS, "confusion_matrix": confusion_matrix(y_test, pred, labels=LABELS).tolist(),
        "classification_report": classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "random_state": SEED,
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"model_name": "social-media-overall-impact", "model_version": "1.0.0", "target": TARGET, "features": X.columns.tolist(), "metrics_file": "metrics.json", "scikit_learn_version": sklearn.__version__, "joblib_version": joblib.__version__}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "best_cv_f1_macro", "best_params"]}, ensure_ascii=False, indent=2))
    print("Artefatos salvos em:", OUT)

if __name__ == "__main__":
    main()
