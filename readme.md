# Modelo indentificar Vogais, Algarismoe e consoante
---

<img width="347" height="305" alt="image-removebg-preview" src="https://github.com/user-attachments/assets/dcddd631-dfa5-41cd-8d23-5fc6c6cc2af9" />


## Estrutura do projeto

```
Licence-Plate-NT/
├── data/
├── notebook/
├──── Convert_folder.ipynb/ 
├── src/
├─── View_Model.py 
├─── Convert_folder.py
├─── Train_Model.py
├─── Install_data.py
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

## Link do dataset
https://www.kaggle.com/datasets/crawford/emnist/data

**Configurações:**
| Parâmetro | Valor |
|-----------|-------|
| Tamanho da imagem | 28×28 pixels |
| Batch size | 128 |
| Épocas máximas | 30 |
| Seed | 42 |

**Arquitetura CNN:**
```
Input (28×28×1)
→ Conv2D(32) + BatchNorm + MaxPool
→ Conv2D(64) + BatchNorm + MaxPool
→ Conv2D(128) + BatchNorm + MaxPool
→ Flatten
→ Dense(256) + Dropout(0.4)
→ Dense(36, softmax)
```

- 3 blocos convolucionais com filtros crescentes (32→64→128)
- `BatchNormalization` estabiliza o treino e acelera a convergência
- `Dropout(0.4)` previne overfitting
- Saída com `softmax` para probabilidade por classe

**Callbacks:**
- `EarlyStopping(patience=5)` — para o treino se a val_accuracy não melhorar por 5 épocas consecutivas e restaura os melhores pesos
- `ReduceLROnPlateau(factor=0.5, patience=3)` — reduz o learning rate pela metade quando estagna por 3 épocas
- `ModelCheckpoint` — salva automaticamente o melhor modelo em `data/model.keras`

**Loss:** `sparse_categorical_crossentropy` — compatível com labels inteiros e com `class_weight`.

---

## Dataset usado
| Split | Imagens |
|-------|---------|
| Treino | 529.006 |
| Teste | 88.299 |
| Total | 617.305 |
