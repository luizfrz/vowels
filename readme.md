# Modelo indentificar Vogais e Algarismo
---

## Estrutura do projeto

```
Licence-Plate-NT/
├── notebook/
├──── Convert_folder.ipynb/
├── src/
├─── main.py
└── requirements.txt
```
## Como usar

### 1. Instalar dependências

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Link do dataset
https://www.kaggle.com/datasets/crawford/emnist/data

Converte o dataset EMNIST (formato CSV) em imagens PNG organizadas por classe.

**O que faz:**
- Lê o arquivo de mapeamento `emnist-byclass-mapping.txt` que associa índices numéricos a caracteres ASCII
- Filtra apenas as 36 classes relevantes: dígitos `0-9` e letras maiúsculas `A-Z`
- Para cada linha do CSV de treino e teste, extrai os 784 pixels (28×28), monta a imagem e corrige a orientação do EMNIST (que vem transposta e espelhada)
- Salva cada imagem em `data/dataset/train/<classe>/` ou `data/dataset/test/<classe>/`

**Correção de orientação:**
O EMNIST armazena as imagens transpostas e espelhadas horizontalmente. O script aplica `TRANSPOSE` e `FLIP_LEFT_RIGHT` para corrigir.

**Resultado:** 529.006 imagens de treino e 88.299 de teste distribuídas em 36 pastas.
---

### `train.py`

Treina uma CNN para classificar os 36 caracteres (0-9, A-Z).

**Configurações:**
| Parâmetro | Valor |
|-----------|-------|
| Tamanho da imagem | 28×28 pixels |
| Batch size | 128 |
| Épocas máximas | 30 |
| Seed | 42 |

**Pipeline de dados:**
- Carrega as imagens com `image_dataset_from_directory` diretamente das pastas
- Aplica data augmentation apenas no treino: rotação (±10°), translação (10%), zoom (10%)
- Normaliza os pixels para o intervalo [0, 1]
- Usa `.prefetch(AUTOTUNE)` para otimizar o carregamento em paralelo

**Desbalanceamento de classes:**
O dataset EMNIST é desbalanceado — dígitos têm ~34k amostras cada, enquanto letras como `B`, `G`, `Q` têm menos de 4k. Sem correção, o modelo vicia nas classes majoritárias.

A solução é calcular `class_weight` com `compute_class_weight("balanced")` do scikit-learn, que aumenta o peso das classes minoritárias durante o treino.

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

## Dataset

O projeto usa o **EMNIST ByClass**, um dataset de caracteres manuscritos com 814.255 imagens de 62 classes (0-9, a-z, A-Z). O `convert.py` filtra apenas as 36 classes relevantes para placas brasileiras.

| Split | Imagens |
|-------|---------|
| Treino | 529.006 |
| Teste | 88.299 |
| Total | 617.305 |

---
