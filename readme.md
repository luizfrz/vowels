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

## Dataset MIN
| Split | Imagens |
|-------|---------|
| Treino| 529.006|
| Teste | 88.299  |
| Total | 617.305 | 
============================================================

    0:  77.7%  (n=5723)
    1:  92.0%  (n=6261)
    2:  91.5%  (n=5814)
    3:  98.0%  (n=5893)
    4:  92.7%  (n=5562)
    5:  89.5%  (n=5141)
    6:  97.2%  (n=5649)
    7:  98.8%  (n=6054)
    8:  95.6%  (n=5583)
    9:  97.0%  (n=5605)
    A:  94.7%  (n=1055)
    B:  97.2%  (n=640)
    C:  97.3%  (n=1723)
    D:  93.9%  (n=770)
    E:  97.4%  (n=843)
    F:  97.5%  (n=1425)
    G:  93.0%  (n=442)
    H:  97.3%  (n=514)
    I:  49.9%  (n=2026)
    J:  95.2%  (n=622)
    K:  98.2%  (n=379)
    L:  96.6%  (n=798)
    M:  97.9%  (n=1471)
    N:  98.4%  (n=1328)
    O:  48.2%  (n=4108)
    P:  96.8%  (n=1386)
    Q:  97.3%  (n=412)
    R:  96.5%  (n=802)
    S:  91.2%  (n=3478)
    T:  95.9%  (n=1561)
    U:  94.8%  (n=1984)
    V:  94.7%  (n=786)
    W:  98.7%  (n=786)
    X:  96.0%  (n=427)
    Y:  94.6%  (n=790)
    Z:  92.8%  (n=458)