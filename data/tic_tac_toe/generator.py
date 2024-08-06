import random
import csv

def generer_sequence():
    grille = [" " for _ in range(9)]
    joueurs = ["O", "X"]  # O commence toujours
    sequence = []
    
    for tour in range(9):
        joueur_actuel = joueurs[tour % 2]
        cases_disponibles = [i for i, case in enumerate(grille) if case == " "]
        
        if not cases_disponibles:
            break
        
        choix = random.choice(cases_disponibles)
        grille[choix] = joueur_actuel
        sequence.append(f"{joueur_actuel}{choix + 1}")
        
        if verifier_victoire(grille, joueur_actuel):
            break
    
    return " ".join(sequence)

def verifier_victoire(grille, joueur):
    combinaisons_gagnantes = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
        [0, 4, 8], [2, 4, 6]  # Diagonales
    ]
    return any(all(grille[i] == joueur for i in combo) for combo in combinaisons_gagnantes)

def generer_parties(nombre_parties, nom_fichier):
    with open(nom_fichier, 'w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["transcript"])  # En-tête de la colonne
        
        for _ in range(nombre_parties):
            partie = generer_sequence()
            writer.writerow([partie])
    
    print(f"{nombre_parties} parties ont été générées et sauvegardées dans {nom_fichier}")

# Utilisation du script
nombre_parties = 100  # Vous pouvez changer ce nombre
nom_fichier = "parties_tic_tac_toe.csv"
generer_parties(nombre_parties, nom_fichier)