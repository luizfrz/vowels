import os
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.keras")
IMG_SIZE = 28
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
VOWELS = set("AEIOU")

model = tf.keras.models.load_model(MODEL_PATH)

def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)

def preprocess(img_array):
    img = Image.fromarray(img_array).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    if arr.std() > 1e-3 and arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return arr


def classify(img_array):
    arr = preprocess(img_array)

    debug = (arr[0, :, :, 0] * 255).astype(np.uint8)
    Image.fromarray(debug).save("debug.png")

    probs = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(probs))
    top = np.argsort(probs)[::-1][:10]

    return {
        "caractere": CLASSES[idx],
        "confianca": float(probs[idx]),
        "tipo": "Algarismo" if CLASSES[idx].isdigit() else "Letra",
        "subtipo": None if CLASSES[idx].isdigit() else ("Vogal" if CLASSES[idx] in VOWELS else "Consoante"),
    }   

BG = "#060606"
FG = "#ffffff"
ACCENT = "#2160c7"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"

root = tk.Tk()
root.title("Classificador Vogais e Algarismo")
root.configure(bg=BG)
root.resizable(False, False)

tk.Label(
    root, text="Classificador de Caractere", font=("Arial", 14, "bold"),
    bg=BG, fg=ACCENT
).grid(row=0, column=0, columnspan=2, pady=(16, 8))

tk.Label(root, text="Imagem (28x28)", font=("Arial", 10), bg=BG, fg=FG).grid(
    row=1, column=0, columnspan=2
)
image_lbl = tk.Label(root, bg=BG)
image_lbl.grid(row=2, column=0, columnspan=2, pady=8)

char_var = tk.StringVar(value="-")
tk.Label(root, text="Caractere previsto", font=("Arial", 10), bg=BG, fg=FG).grid(
    row=3, column=0, columnspan=2
)
char_lbl = tk.Label(root, textvariable=char_var, font=("Arial", 40, "bold"), bg=BG, fg=ACCENT)
char_lbl.grid(row=4, column=0, columnspan=2, pady=(0, 4))

conf_var = tk.StringVar(value="")
tk.Label(root, textvariable=conf_var, font=("Arial", 10), bg=BG, fg=FG).grid(
    row=5, column=0, columnspan=2, pady=(0, 10)
)

tipo_var = tk.StringVar(value="")
tipo_lbl = tk.Label(root, textvariable=tipo_var, font=("Arial", 13, "bold"), bg=BG)
tipo_lbl.grid(row=6, column=0, columnspan=2)

subtipo_var = tk.StringVar(value="")
subtipo_lbl = tk.Label(root, textvariable=subtipo_var, font=("Arial", 12), bg=BG)
subtipo_lbl.grid(row=7, column=0, columnspan=2, pady=(0, 16))


def process(path):
    img_array = load_image(path)
    result = classify(img_array)

    preview = Image.fromarray(img_array).resize((160, 160), Image.NEAREST)
    preview_tk = ImageTk.PhotoImage(preview)
    image_lbl.configure(image=preview_tk)
    image_lbl.image = preview_tk

    char_var.set(result["caractere"])
    conf_var.set(f"Confianca: {result['confianca'] * 100:.1f}%")

    tipo_lbl.configure(fg=ACCENT if result["tipo"] == "Letra" else GREEN)
    tipo_var.set(f"Tipo: {result['tipo']}")

    if result["subtipo"]:
        subtipo_lbl.configure(fg=YELLOW)
        subtipo_var.set(f"Subtipo: {result['subtipo']}")
    else:
        subtipo_var.set("")


def upload():
    path = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")]
    )
    if path:
        process(path)


tk.Button(
    root, text="Carregar Imagem", command=upload,
    font=("Arial", 11), bg=ACCENT, fg="#1e1e2e",
    padx=14, pady=7, relief="flat"
).grid(row=8, column=0, columnspan=2, pady=(0, 20))

root.mainloop()