# nn — Rede Neural simples (Python)

> Implementação didática de uma pequena framework de redes neurais, sendo abordadas operações de uma Fully Connected e de uma Convolucional
---

## Visão geral

Este repositório contém uma implementação simples e educativa de uma rede neural com camadas, forward/backward, salvamento/carregamento de pesos e um conjunto de testes automatizados (pytest). O objetivo é servir como base para estudos, experimentos e trabalhos de faculdade.

Principais funcionalidades

* Camadas (layers) básicas (Dense/CNN, ativação, etc.)
* Estrutura `Model` com `add`, `forward`, `backward` e utilitários
* Salvamento e carregamento de pesos (`save_weights` / `load_weights`)
* Grad clipping e medidas de estabilidade numérica
* Testes com `pytest`

---

## Status

**Em desenvolvimento.** Funcionalidades principais implementadas; testes e exemplos cobrem casos básicos. Use este README como ponto de partida e atualize conforme o projeto evolui.

---

## Requisitos

* Python 3.8+ (testado com 3.11/3.13)
* pip
* Dependências (opcionais): `numpy`, `pytest` (veja `requirements.txt`)

---

## Instalação (local / recomendada)

1. Crie e ative um ambiente virtual (opcional mas recomendado):

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
.venv\Scripts\activate
```

2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Instale o pacote localmente (modo desenvolvimento) para evitar problemas de import durante os testes:

```bash
pip install -e .
```

> Alternativa rápida (não recomendada para repositórios): exporte `PYTHONPATH` para o diretório do projeto, por exemplo `export PYTHONPATH=.`, ou rode `pytest` a partir do diretório raiz com `python -m pytest`.

---

## Estrutura sugerida de diretórios

```
nn/                # raiz do repositório
├─ src/             # código fonte
│  ├─ framework/          
│  ├─ __init__.py
│  ├─ model.py      # modelo
│  ├─ layers.py     # layers, activation, losses, optimizer etc
│  ├─ tests/        # testes pytest
├─ pyproject.toml
├─ README.md
```

**Observação:** adotar o layout `src/` e instalar com `pip install -e .` reduz bastante problemas de import em testes e em desenvolvimento local.

---

## Como rodar

### Testes

```bash
# a partir da raiz do projeto
pytest -q
```

Se o pytest não encontrar os módulos (ImportError), certifique-se de que:

* Você instalou o package (`pip install -e .`) **ou**
* Está rodando pytest a partir da raiz do projeto com `PYTHONPATH=.`, ou
* Há um `__init__.py` adequado e o layout é de pacote.

### Exemplo rápido

```python
from .model import Model
from .layers import Dense, ReLU

model = Model()
model.add(Dense(784, 128))
model.add(ReLU())
# ... definir loss, otimização, etc.

# Treinar/rodar forward
# model.fit(...)
```

---

## Boas práticas e dicas para o desenvolvimento

* Use `pip install -e .` durante o desenvolvimento para que `import nn` funcione de qualquer lugar.
* Se você precisa importar arquivos de um diretório anterior sem empacotar, prefira explicitar o caminho no `sys.path` (somente para testes rápidos) ou, melhor, transformar o diretório em um package com `__init__.py` ou usar o layout `src/`.

---

## Exemplo de configuração CI (GitHub Actions)

```yaml
name: Python package
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.11]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install -r requirements.txt
      - run: pytest -q
```

## Contato

Se quiser, me marque nas issues ou envie uma mensagem no GitHub.

