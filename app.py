"""Aplicação Streamlit para inferência do modelo Overall_Impact."""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # permite executar localmente sem Hub instalado
    hf_hub_download = None

ROOT = Path(__file__).resolve().parent

def get_setting(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, os.getenv(key, default)))
    except Exception:
        return os.getenv(key, default)

LOCAL_MODEL = ROOT / "artifacts" / "model.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="Impacto das Redes Sociais", page_icon="📊", layout="centered")
st.title("Classificação do impacto das redes sociais")
st.caption("Projeto acadêmico — Desafio 04 | Machine Learning em nuvem")
st.warning(
    "Demonstração acadêmica. A previsão não é diagnóstico psicológico, não estabelece causalidade "
    "e não deve ser usada para decisões individuais de alto impacto."
)

@st.cache_resource
def load_model():
    repo_id = get_setting(
        "HF_MODEL_REPO_ID",
        "myllac/desafio04-impacto-redes-sociais"
    ).strip()

    if repo_id:
        if hf_hub_download is None:
            raise RuntimeError("Instale huggingface_hub conforme requirements.txt.")

        token = get_setting("HF_TOKEN") or None

        path = hf_hub_download(
            repo_id=repo_id,
            filename="model.joblib",
            repo_type="model",
            token=token
        )

        return joblib.load(path), f"Hugging Face Hub: {repo_id}"

    if LOCAL_MODEL.exists():
        return joblib.load(LOCAL_MODEL), "arquivo local"

    raise FileNotFoundError(
        "Modelo não encontrado. Verifique a configuração HF_MODEL_REPO_ID."
    )

with st.expander("Sobre o projeto"):
    st.write(
        "O modelo estima a classe Overall_Impact (Beneficial, Neutral ou Negative) "
        "a partir de características de uso de redes sociais, sono, estresse e desempenho acadêmico. "
        "A coluna Student_ID não é utilizada como variável preditora."
    )

st.subheader("Informe as características")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Idade", min_value=13, max_value=100, value=20)
        gender = st.selectbox("Gênero", ["Female", "Male", "Non-Binary", "Prefer not to say"])
        academic = st.selectbox("Nível acadêmico", ["High School", "Undergraduate", "Postgraduate"])
        platform = st.selectbox("Plataforma principal", ["Instagram", "LinkedIn", "Reddit", "Snapchat", "TikTok", "X (Twitter)", "YouTube"])
        device = st.selectbox("Dispositivo", ["Smartphone", "Laptop/PC", "Tablet"])
        late_night = st.selectbox("Uso tarde da noite?", [True, False], format_func=lambda x: "Sim" if x else "Não")
    with c2:
        daily = st.number_input("Horas diárias de uso", 0.0, 24.0, 5.0, 0.1)
        weekend = st.number_input("Horas extras no fim de semana", 0.0, 24.0, 1.0, 0.1)
        sleep = st.number_input("Duração do sono (horas)", 0.0, 24.0, 7.0, 0.1)
        sleep_quality = st.slider("Qualidade do sono (1 a 5)", 1, 5, 3)
        comparison = st.selectbox("Frequência de comparação social", ["Never", "Rarely", "Sometimes", "Frequently", "Always"])
        stress = st.number_input("Pontuação de estresse percebido", 0.0, 100.0, 15.0, 0.5)
        mental = st.slider("Índice de saúde mental", 0, 100, 70)
        gpa = st.number_input("GPA / desempenho acadêmico", 0.0, 4.0, 3.0, 0.01)
    submitted = st.form_submit_button("Classificar impacto", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Age": age, "Gender": gender, "Academic_Level": academic,
        "Primary_Platform": platform, "Daily_Usage_Hours": daily,
        "Weekend_Extra_Hours": weekend, "Device_Type": device,
        "Sleep_Duration_Hours": sleep, "Sleep_Quality_Score": sleep_quality,
        "Late_Night_Usage": late_night,
        "Social_Comparison_Frequency": comparison,
        "Perceived_Stress_Score": stress, "Mental_Health_Index": mental,
        "Academic_Performance_GPA": gpa,
    }])
    start = time.perf_counter()
    try:
        model, model_source = load_model()
        expected = list(getattr(model, "feature_names_in_", row.columns))
        missing = [column for column in expected if column not in row.columns]
        if missing:
            raise ValueError(f"Campos ausentes para o modelo: {missing}")
        row = row.reindex(columns=expected)
        prediction = str(model.predict(row)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = dict(zip(model.classes_, model.predict_proba(row)[0].tolist()))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logging.info(json.dumps({"event": "prediction", "prediction": prediction, "latency_ms": latency_ms, "model_source": model_source}, ensure_ascii=False))
        st.success(f"Classe prevista: **{prediction}**")
        st.caption(f"Modelo: {model_source} | Latência desta chamada: {latency_ms} ms")
        if probability:
            st.write("Probabilidades estimadas")
            st.bar_chart(pd.Series(probability, name="Probabilidade"))
    except Exception as exc:
        logging.exception("Falha na inferência")
        st.error(f"Não foi possível executar a inferência: {exc}")

st.divider()
st.caption("Monitoramento demonstrativo: eventos de previsão, erros e latência são enviados aos logs da aplicação. A hospedagem gratuita pode reiniciar e não oferece persistência garantida desses logs.")
