# Modelo indentificar Vogais, Algarismo e consoante
---

<img width="347" height="305" alt="image-removebg-preview" src="https://github.com/user-attachments/assets/dcddd631-dfa5-41cd-8d23-5fc6c6cc2af9" />

## Estrutura do projeto
```
Vogais/
├── data/
├── notebook/ ➔ Código comentado
├──── instal_dataset.ipynb/ 
├──── slipt_data.ipynb/ 
├──── training.ipynb/ 
├── src/ 
├──── instal_dataset.py/ 
├──── slipt_data.py/ 
├──── training.py/ 
├──── view.py/ 
└── requirements.txt
```

## Como usar

### 1. Instalar 

#### Criar ambiente
``` bash
python3 -m venv venv
```
#### Instala  dependências
```bash 
pip install -r requirements.txt
```

## Dataset 
https://www.kaggle.com/datasets/crawford/emnist/data

**Configurações:**
| Parâmetro | Valor |
|-----------|-------|
| Tamanho da imagem | 28×28 pixels |
| Batch size | 128 |
| Épocas máximas | 30 |
| Seed | 42 |

## Dataset EMNIST
| Split | Imagens |
|-------|---------|
| Treino| 529.006|
| Teste | 88.299  |
| Total | 617.305 | 

## Visão Geral do Pipeline Completo

```mermaid
flowchart LR
    A[(EMNIST CSV\n+ Mapping)] -->|split_data.py| B[(data/emnist_source_files\ntrain/ · test/)]
    B -->|training.py| C[(data/model\nmodel.keras)]
    C -->|view.py| D([Interface Tkinter\nClassificação em tempo real])
```
