import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder
import requests
import chardet

# Dictionnaire de conversion fourni
meta = {
    'stoi': {' ': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'X': 10, 'O': 11},
    'itos': {0: ' ', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'X', 11: 'O'}
}

dtype = np.uint8  # 32 tokens seulement dans le vocabulaire des LLMs pour les échecs

# Authentification auprès de Hugging Face
hf_token = "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"
HfFolder.save_token(hf_token)
api = HfApi()

# Téléchargement du fichier CSV depuis Hugging Face
dataset_path = "Zual/tic-tac-toe"
file_path = "26830_tictactoe.csv"
local_file_path = "26830_tictactoe.csv"

if not os.path.exists(local_file_path):
    url = f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{file_path}"
    response = requests.get(url)
    with open(local_file_path, 'wb') as f:
        f.write(response.content)
    print(f"Fichier '{local_file_path}' téléchargé.")

# Détection de l'encodage du fichier
with open(local_file_path, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']

print(f"Encodage détecté : {detected_encoding}")

# Chargement du dataset à partir du fichier CSV téléchargé
encodings_to_try = [detected_encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

for encoding in encodings_to_try:
    try:
        data = pd.read_csv(local_file_path, encoding=encoding)
        print(f"Lecture réussie avec l'encodage : {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Échec de lecture avec l'encodage : {encoding}")
else:
    raise ValueError("Impossible de lire le fichier avec les encodages essayés.")

data = data.drop(0)  # Supprimer la première ligne contenant "transcript"

def process_line(line, meta, vector_size=27):
    vector = np.zeros(vector_size, dtype=dtype)
    for i, char in enumerate(line.strip()):
        if i >= vector_size:
            break
        vector[i] = meta['stoi'].get(char, 0)  # Utiliser 0 si le caractère n'est pas trouvé
    return vector

# Traitement des données
batches = []
for _, row in tqdm(data.iterrows(), total=len(data), desc="Traitement des lignes"):
    text = row['transcript']
    batch = process_line(text, meta)
    batches.append(batch)

batches = np.array(batches)

# Sauvegarde des ensembles dans des fichiers binaires
train_ratio = 0.01  # 1% des parties possibles de tic tac toe pour l'entraînement pour voir le grok !
split_index = int(len(batches) * train_ratio)
train_batches = batches[:split_index]
val_batches = batches[split_index:]

train_batches.tofile("train.bin")
val_batches.tofile("val.bin")

print(f"Ensemble d'entraînement sauvegardé dans 'train.bin'")
print(f"Ensemble de validation sauvegardé dans 'val.bin'")

# Affichage des 50 premières lignes du CSV pour vérification
print("50 premières lignes du CSV:")
print(data.head(50))

# Informations sur les ensembles de données
print(f"\nNombre total d'exemples : {len(batches)}")
print(f"Nombre d'exemples d'entraînement : {len(train_batches)}")
print(f"Nombre d'exemples de validation : {len(val_batches)}")

import numpy as np

def load_and_print_batches(filename, start_batch=0, end_batch=50, batch_size=27):
    # Charger le fichier binaire
    data = np.fromfile(filename, dtype=np.uint8)

    # Calculer le nombre total de batches
    total_batches = len(data) // batch_size

    # Limiter l'intervalle de batches à afficher si nécessaire
    start_batch = max(0, start_batch)
    end_batch = min(end_batch, total_batches)

    # Afficher les batches dans l'intervalle spécifié
    for i in range(start_batch, end_batch):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = data[batch_start:batch_end]
        print(f"Batch {i+1}: {batch}")

# Exemple d'utilisation : afficher les batches de 10 à 20
load_and_print_batches("train.bin", start_batch=1, end_batch=10)
