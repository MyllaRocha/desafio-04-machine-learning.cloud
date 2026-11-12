---
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
- tabular-classification
- scikit-learn
---

# Social Media Overall Impact — modelo acadêmico

Modelo de classificação multiclasse para estimar `Overall_Impact` (Beneficial, Neutral, Negative) a partir de características tabulares de hábitos de uso de redes sociais, sono, estresse e desempenho acadêmico.

## Artefato
`model.joblib` contém um pipeline scikit-learn completo, incluindo imputação, codificação, padronização e classificador. Foi treinado com `src/train.py`.

## Avaliação
- Acurácia: 98,22%
- Precisão macro: 96,40%
- Recall macro: 94,92%
- F1 macro: 95,62%

Partição estratificada 80/20, `random_state=42`. A classe Negative é minoritária; consulte métricas por classe em `metrics.json`.

## Limitações e uso responsável
Resultados dependem da base utilizada e não demonstram causalidade. O modelo não é ferramenta diagnóstica e não deve ser usado para decisões individuais de alto impacto. Verifique licença e procedência dos dados antes de redistribuí-los.
