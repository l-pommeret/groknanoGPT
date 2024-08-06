import pickle

vocab = {
    'vocab_size': 12,
    'itos': {0: ' ', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'X', 11: 'O'},
    'stoi': {' ': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'X': 10, 'O': 11}
}

# Écriture du dictionnaire dans un fichier pickle
with open('meta.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Le fichier meta.pkl a été créé avec succès.")

# Pour vérifier le contenu (optionnel)
with open('meta.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
print("Contenu chargé :", loaded_vocab)