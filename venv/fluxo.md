┌──────────────────────────────┐
│           INÍCIO             │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Importar bibliotecas         │
│ os, json, numpy, TensorFlow  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Configurar SEED = 42         │
│ e parâmetros do treinamento  │
│ IMG_SIZE = 28                │
│ BATCH_SIZE = 128             │
│ EPOCHS = 20                  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Ler classes de:              │
│ train/ e test/               │
└──────────────┬───────────────┘
               ↓
        ┌─────────────────┐
        │ Classes de      │
        │ train == test?  │
        └───────┬─────────┘
          SIM   │   NÃO
          ↓     │     ↓
          │     │  ┌──────────────────────┐
          │     └─→│ Exibir diferenças    │
          │        │ entre as pastas      │
          │        │ e gerar RuntimeError │
          │        └──────────┬───────────┘
          │                   ↓
          │                 FIM
          ↓
┌──────────────────────────────┐
│ Contar amostras por classe   │
│ no conjunto de treinamento  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Encontrar a menor quantidade │
│ de imagens entre as classes  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ UNDERSAMPLING                │
│                              │
│ Para cada classe:            │
│ • carregar imagens           │
│ • limitar para min_count     │
│ • atribuir o índice da classe│
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Combinar todas as classes    │
│ com pesos iguais             │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Embaralhar (shuffle)         │
│ Agrupar em batches           │
│ Usar prefetch automático     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Criar val_data               │
│                              │
│ Imagens do TEST              │
│ 28x28                         │
│ Escala de cinza               │
│ Classes originais            │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Calcular:                    │
│ NUM_CLASSES                  │
│ steps_per_epoch              │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Construir modelo CNN         │
│                              │
│ Input 28x28x1                │
│       ↓                      │
│ Rescaling                    │
│       ↓                      │
│ Data Augmentation            │
│ • Rotation                   │
│ • Translation                │
│ • Zoom                       │
│ • Contrast                   │
│ • Gaussian Noise             │
│       ↓                      │
│ Conv2D 32                    │
│ BatchNormalization           │
│ MaxPooling                   │
│       ↓                      │
│ Conv2D 64                    │
│ BatchNormalization           │
│ MaxPooling                   │
│       ↓                      │
│ Conv2D 128                   │
│ BatchNormalization           │
│ MaxPooling                   │
│       ↓                      │
│ Flatten                      │
│       ↓                      │
│ Dense 256                    │
│       ↓                      │
│ Dropout 40%                  │
│       ↓                      │
│ Dense NUM_CLASSES            │
│ Softmax                      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Compilar modelo              │
│                              │
│ Optimizer: Adam              │
│ Loss: Sparse Categorical     │
│       Crossentropy           │
│ Métrica: Accuracy            │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Configurar Callbacks         │
│                              │
│ • ProgressCallback           │
│ • EarlyStopping              │
│ • ReduceLROnPlateau          │
│ • ModelCheckpoint            │
└──────────────┬───────────────┘
               ↓
        ┌─────────────────┐
        │ TREINAMENTO     │
        │ até 20 épocas   │
        └───────┬─────────┘
                ↓
       ┌─────────────────────┐
       │ EarlyStopping       │
       │ ou fim das épocas?  │
       └─────────┬───────────┘
           NÃO   │    SIM
           └─────┤      ↓
                 │   ┌──────────────────┐
                 │   │ Restaurar pesos  │
                 │   │ da melhor época  │
                 │   └────────┬─────────┘
                 │            ↓
                 └────────────┘
                              ↓
┌──────────────────────────────┐
│ Carregar val_data            │
│ e realizar predições         │
│ com model.predict()          │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Obter y_true                 │
│ (classes reais)              │
│ e y_pred                     │
│ (classes previstas)          │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Calcular acurácia de         │
│ cada classe                  │
└──────────────┬───────────────┘
               ↓
        ┌─────────────────┐
        │ Acurácia da     │
        │ classe = 0%?    │
        └───────┬─────────┘
          SIM   │   NÃO
          ↓     │     ↓
┌───────────────┐ │ ┌────────────────┐
│ Exibir        │ │ │ Exibir         │
│ "nunca acertou"│ │ │ acurácia       │
└───────┬───────┘ │ └───────┬────────┘
        └─────────┴──────────┘
                  ↓
┌──────────────────────────────┐
│ Calcular acurácia geral      │
│ (y_true == y_pred)           │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Exibir resultados:           │
│ • Acurácia geral             │
│ • Melhor val_accuracy        │
│ • Acurácia por classe        │
│ • Quantidade por classe      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Salvar / informar caminho    │
│ do melhor modelo             │
│ MODEL_OUT                    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│             FIM              │
└──────────────────────────────┘
