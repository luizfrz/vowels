import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "data", "model.keras")
HISTORY_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "history.json")

IMG_SIZE   = 28
BATCH_SIZE = 128
EPOCHS     = 20

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

classes      = sorted(os.listdir(TRAIN_DIR))
test_classes = sorted(os.listdir(TEST_DIR))


if classes != test_classes:
    only_in_train = set(classes) - set(test_classes)
    only_in_test  = set(test_classes) - set(classes)
    msg = ["Pastas de train/ e test/ não coincidem!"]
    if only_in_train:
        msg.append(f"  Só em train/: {sorted(only_in_train)}")
    if only_in_test:
        msg.append(f"  Só em test/:  {sorted(only_in_test)}")
    raise RuntimeError("\n".join(msg))

counts    = {c: len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in classes}
min_count = min(counts.values())

print(f"Classes: {len(classes)}")
print(f"Amostras por classe (antes): min={min_count}  max={max(counts.values())}")
print(f"Amostras por classe (após undersampling): {min_count}")
print(f"Total treino: {min_count * len(classes):,}\n")

class_datasets = []
for i, c in enumerate(classes):
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=[c],
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=None,
        seed=SEED,
    )
    ds = ds.take(min_count)
    ds = ds.map(lambda x, _, i=i: (x, tf.cast(i, tf.int32)))
    class_datasets.append(ds)

train_data = (
    tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / len(classes)] * len(classes),
        seed=SEED,
    )
    .shuffle(10000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_data = (
    tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=classes,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    .prefetch(tf.data.AUTOTUNE)
)

NUM_CLASSES     = len(classes)
steps_per_epoch = (min_count * NUM_CLASSES) // BATCH_SIZE

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Rescaling(1.0 / 255),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.3),
    layers.GaussianNoise(0.05),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

class ProgressCallback(callbacks.Callback):

    def on_train_begin(self, logs=None):
        print(f"\n{'='*60}")
        print(f"  Iniciando treino — {EPOCHS} épocas máximas")
        print(f"{'='*60}\n")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Época {epoch+1}/{EPOCHS}")

    def on_epoch_end(self, epoch, logs=None):
        acc      = logs.get("accuracy", 0) * 100
        val_acc  = logs.get("val_accuracy", 0) * 100
        loss     = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        lr       = float(self.model.optimizer.learning_rate)

        filled = int(30 * val_acc / 100)
        bar    = "█" * filled + "░" * (30 - filled)

        print(f"  [{bar}] {val_acc:.1f}%")
        print(f"  acc={acc:.2f}%  val_acc={val_acc:.2f}%  loss={loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.6f}")
        print()

    def on_train_end(self, logs=None):
        print(f"{'='*60}")
        print(f"  Treino finalizado!")
        print(f"{'='*60}\n")

cbs = [
    ProgressCallback(),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=cbs,
    verbose=0,
)

print(f"Modelo salvo em: {MODEL_OUT}")
print(f"Melhor val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

with open(HISTORY_OUT, "w") as f:
    json.dump(history.history, f, indent=2)
print(f"Histórico salvo em: {HISTORY_OUT}")
 
print(f"\n{'='*60}")
print("  Avaliação por classe (val_data)")
print(f"{'='*60}\n")

y_true = []
y_pred = []
for x_batch, y_batch in val_data:
    probs = model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_true.extend(y_batch.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

predicted_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
never_predicted  = [classes[i] for i in range(NUM_CLASSES) if predicted_counts[i] == 0]

for i, c in enumerate(classes):
    mask = (y_true == i)
    n = mask.sum()
    if n == 0:
        continue
    acc_classe = (y_pred[mask] == i).mean() * 100
    flag = "  < nunca acertou" if acc_classe == 0 else ""
    print(f"  {c:>3}: {acc_classe:5.1f}%  (n={n}){flag}")

overall_acc = (y_true == y_pred).mean() * 100
print(f"\nAcurácia geral (recalculada): {overall_acc:.2f}%")