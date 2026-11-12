# Publicar este projeto no GitHub

## 1. Criar o repositório
- Entre no GitHub e clique em **New repository**.
- Nome sugerido: `desafio-04-machine-learning-cloud`.
- Descrição: `Classificação do impacto das redes sociais — Desafio 04`.
- Para facilitar a implantação, escolha **Public** (confira antes a licença do dataset e não inclua dados pessoais ou segredos).
- Não marque a opção de criar README, `.gitignore` ou licença: estes arquivos já estão preparados aqui.

## 2. Enviar pelo navegador (mais simples)
- Abra o repositório recém-criado.
- Clique em **Add file → Upload files**.
- Extraia o ZIP do kit e arraste o conteúdo da pasta `Desafio04_GitHub` para a página do GitHub.
- Faça o commit com a mensagem `Estrutura inicial do Desafio 04`.
- Confira se `README.md`, `app.py`, `requirements.txt`, `data/`, `notebooks/`, `src/` e `docs/` aparecem no repositório.

## 3. Alternativa usando Git no computador
Dentro da pasta do projeto:

```bash
git init
git add .
git commit -m "Estrutura inicial do Desafio 04"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/desafio-04-machine-learning-cloud.git
git push -u origin main
```

Troque `SEU-USUARIO` pelo seu usuário GitHub e use a URL real do repositório. Não compartilhe senha ou token no chat.
