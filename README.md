# Desafio 04 — Serviços de Computação em Nuvem para Machine Learning

**Projeto:** classificação do impacto das redes sociais na vida de estudantes (`Overall_Impact`).

Este repositório organiza o fluxo de preparação, treinamento, avaliação e inferência iniciado na Sprint 03. A aplicação web foi preparada para publicação no Streamlit Community Cloud; o artefato do modelo pode ser armazenado em um repositório público do Hugging Face Hub.

> **Status:** estrutura de código preparada. A publicação no GitHub, o treinamento final, o envio do modelo ao Hub e a implantação no Streamlit ainda precisam ser executados e comprovados com evidências reais.

## Arquitetura proposta

1. **Google Colab:** ambiente hospedado para executar o notebook e/ou o script de treinamento.
2. **Hugging Face Hub:** armazenamento versionado do artefato `model.joblib` (repositório do tipo Model).
3. **GitHub:** versionamento do código, notebook e documentação.
4. **Streamlit Community Cloud:** hospedagem da interface de inferência.
5. **Logs da aplicação:** registro de chamadas, erros e latência via logs do Streamlit. Isso é monitoramento em nível de aplicação, não equivale ao conjunto gerenciado de métricas do Amazon CloudWatch.

## Estrutura

```text
Desafio04_GitHub/
├── app.py                         # Interface de inferência Streamlit
├── requirements.txt               # Dependências da aplicação
├── data/
│   └── Social_media_impact_on_life.csv
├── notebooks/
│   └── Desafio04_ML_Cloud.ipynb   # Notebook de treinamento e avaliação
├── src/
│   ├── train.py                   # Treinamento reproduzível
│   └── inference.py               # Funções de inferência
├── artifacts/                     # Gerado após treinamento (não versionar modelo)
├── docs/
│   └── Arquitetura_Proposta_Desafio04.png
└── evidence/                      # Capturas de tela reais da execução/publicação
```

## Execução local ou no Colab

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

O treinamento gera `artifacts/model.joblib`, `artifacts/metrics.json` e `artifacts/manifest.json`. O arquivo do modelo está no `.gitignore` para evitar versionar binários no GitHub; envie-o separadamente ao repositório de modelo no Hugging Face Hub. O `requirements.txt` fixa versões compatíveis com o artefato gerado neste pacote; mantenha a versão do scikit-learn igual à usada no treinamento para evitar incompatibilidade de serialização.

## Publicação do modelo no Hugging Face Hub

1. Crie um repositório **Model** público no Hugging Face Hub.
2. Faça upload de `artifacts/model.joblib` e, opcionalmente, `artifacts/manifest.json` e `artifacts/metrics.json`.
3. Guarde o identificador no formato `usuario/nome-do-modelo`.
4. Não coloque tokens no código nem neste repositório. Para repositório público, não é necessário token de leitura.

## Publicação no Streamlit Community Cloud

1. Envie este repositório ao GitHub.
2. No Streamlit Community Cloud, conecte o GitHub e crie um app apontando para `app.py`.
3. Em **App settings → Secrets**, adicione:

```toml
HF_MODEL_REPO_ID = "usuario/nome-do-modelo"
```

4. Se o repositório do modelo for privado, configure `HF_TOKEN` nos Secrets. Nunca publique esse token.
5. Abra o link do app, faça uma previsão de teste e registre capturas de tela em `evidence/`.

## Monitoramento e evidências

A aplicação escreve eventos estruturados nos logs do processo com tipo do evento, classe prevista e latência. Registre evidências de:

- repositório GitHub e sua estrutura;
- execução do notebook/script e métricas efetivamente obtidas;
- repositório do modelo no Hugging Face Hub e arquivos armazenados;
- configuração do app no Streamlit e URL pública;
- teste de inferência e logs disponíveis.

Não invente capturas de tela ou resultados: anexe somente evidências de operações realmente concluídas. A hospedagem gratuita pode hibernar apps sem tráfego e não garante persistência de arquivos/logs locais.

## Resultados de referência já registrados

Uma execução independente anterior, com divisão estratificada 80/20 e `random_state=42`, registrou acurácia de **98,22%**, precisão macro de **96,40%**, recall macro de **94,92%** e F1 macro de **95,62%**. Execute novamente `python src/train.py` ou o notebook e atualize os resultados com a saída real do ambiente final.

## Limitações e uso responsável

- A classe `Negative` é minoritária; não avalie o modelo apenas pela acurácia. Consulte F1 macro e métricas por classe.
- O modelo identifica padrões da base e não demonstra causalidade.
- Indicadores de estresse e saúde mental não devem ser usados para diagnóstico ou decisões individuais de alto impacto.
- Antes de publicar o CSV, confirme a licença e a permissão de redistribuição da fonte original.
- Google Colab é um ambiente de notebook hospedado, não um serviço gerenciado de treinamento equivalente em todos os aspectos ao SageMaker. Streamlit é uma aplicação hospedada que realiza inferência, não um endpoint gerenciado equivalente ao SageMaker Endpoint. Descreva essas diferenças no relatório.

## Referências oficiais

- Google Colab: https://colab.research.google.com/
- Hugging Face Hub: https://huggingface.co/docs/hub/index
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- GitHub Docs: https://docs.github.com/


## Correções aplicadas nesta revisão
- O notebook agora aceita o CSV diretamente (não tenta abrir CSV como ZIP).
- A instalação padrão do Colab não instala `boto3`/`sagemaker`; as etapas AWS permanecem opcionais.
- O script de treinamento verifica a existência do CSV e das colunas obrigatórias.
- A aplicação alinha as colunas de entrada à ordem esperada pelo pipeline.
- O modelo foi treinado novamente a partir do CSV incluído e o artefato foi gerado e testado nesta revisão.

### Teste rápido do artefato

```bash
pip install pytest
pytest -q
```
