# Como gerar `model.joblib`

1. No Colab, abra `notebooks/Desafio04_ML_Cloud.ipynb`.
2. Execute as células em ordem, do início até a seção 8.
3. Quando solicitado, envie `Social_media_impact_on_life.csv` (arquivo CSV; não ZIP).
4. Ao final, o notebook cria `artifacts/model.joblib`.
5. Para baixar:
```python
from google.colab import files
files.download("artifacts/model.joblib")
```
6. Envie `model.joblib`, `manifest.json` e `metrics.json` para um repositório Hugging Face do tipo Model.

Se preferir executar o script:
```bash
pip install -r requirements.txt
python src/train.py
```
