import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(BASE_DIR, "data", "emnist_source_files")

TRAIN_CSV = os.path.join(DATA_DIR, "emnist-byclass-train.csv")
TEST_CSV = os.path.join(DATA_DIR, "emnist-byclass-test.csv")
MAPPING = os.path.join(DATA_DIR, "emnist-byclass-mapping.txt")

os.makedirs(OUTPUT, exist_ok=True)

mapping = {}

with open(MAPPING) as f:
    for line in f:
        idx, ascii_code = line.strip().split()
        idx = int(idx)
        ascii_code = int(ascii_code)

        mapping[idx] = chr(ascii_code)

valid_classes = {}

new_label = 0

for old, char in mapping.items():

    if char.isdigit():

        valid_classes[old] = {
            "char": char,
            "label": new_label
        }

        new_label += 1

    elif char.isupper():

        valid_classes[old] = {
            "char": char,
            "label": new_label
        }

        new_label += 1

print("Classes encontradas:", len(valid_classes))

def process(csv_file, split):

    df = pd.read_csv(csv_file, header=None)

    for row in tqdm(df.itertuples(index=False), total=len(df)):

        old_label = row[0]

        if old_label not in valid_classes:
            continue

        char = valid_classes[old_label]["char"]

        folder = os.path.join(
            OUTPUT,
            split,
            char
        )

        os.makedirs(folder, exist_ok=True)

        pixels = list(row[1:])

        img = Image.new("L", (28,28))
        img.putdata(pixels)

        img = img.transpose(Image.TRANSPOSE)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = list(img.getdata())
        if sum(arr) / len(arr) > 127:
            arr = [255 - p for p in arr]
            img.putdata(arr)

        filename = os.path.join(
            folder,
            f"{tqdm.format_num(process.counter)}.png"
        )

        img.save(filename)

        process.counter += 1

process.counter = 0

print("Convertendo treino...")
process(TRAIN_CSV, "train")

print("Convertendo teste...")
process(TEST_CSV, "test")

print("Dataset criado com sucesso!")
