# Modelo indentificar Vogais, Algarismo e consoante
---

<img width="347" height="305" alt="image-removebg-preview" src="https://github.com/user-attachments/assets/dcddd631-dfa5-41cd-8d23-5fc6c6cc2af9" />

## Sobre projeto
Uma rede neural criado sobre um desafio lançado do grupo de *Boitatá lab* para indentificar letras e número, indentificar se consoante, vogais ou algarismo tamanho *28x28*.


## Estrutura do projeto
```
Vogais/
├── data/
├─── example/ ➔ exemplos
├─── model/
├─── graph/
├── notebook/ ➔ Código comentado
├──── split_data.ipynb/ 
├──── training.ipynb/ 
├── src/ 
├──── instal_dataset.py/ 
├──── slipt_data.py/ 
├──── training.py/ 
├──── view.py/ 
└── requirements.txt
```

## Como utilizar 

### Efetue clonagem do projeto 
``` bash
git clone https://github.com/luizfrz/vowels
```

#### Criar ambiente
``` bash
python3 -m venv venv
```
#### Ativar ambiente
``` bash
source venv/bin/activate
```
#### Instala  dependências
```bash 
pip install -r requirements.txt
```
#### Faz divisão de train/ teste/
```bash 
python3 split_data.py
```
#### Treinamento do modelo
```bash 
python3 training.py
```
#### Classificação 
```bash 
python3 view.py
```

## Dataset utilizado
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

## Modelo indentificando 
<img width="300" height="400" alt="Screenshot_2026-08-19_15-13-36" src="https://github.com/user-attachments/assets/ea535b60-a5f3-4261-9348-5d87b406c2dd" />
<img width="300" height="400" alt="Screenshot_2026-08-19_15-08-51" src="https://github.com/user-attachments/assets/8cf6125c-f778-47fd-8c2c-5431e4fdfbe3" />


## Visão Geral do Pipeline Completo

```mermaid
flowchart LR
    A[(EMNIST CSV\n+ Mapping)] -->|split_data.py| B[(data/emnist_source_files\ntrain/ · test/)]
    B -->|training.py| C[(data/model\nmodel.keras)]
    C -->|view.py| D([Interface Tkinter\nClassificação em tempo real])
```
