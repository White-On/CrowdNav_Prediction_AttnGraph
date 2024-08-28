#!/bin/bash


# Script Python à lancer
PYTHON_SCRIPT="evaluate_PPO_stable_baseline.py"

# Fonction pour arrêter tous les processus enfants
cleanup() {
    echo "Arrêt des scripts..."
    pkill -P $$
    exit 1
}

# Capturer les signaux de terminaison
trap cleanup SIGINT SIGTERM

# trouve les fichiers de modèle -> le nom commence par MODEL_FILE
ALL_MODELS=$(find . -name "MODEL_FILE*")

# Pour chaque fichier de modèle, on lance le script Python
for MODEL in $ALL_MODELS
do
    echo "Lancement du script Python avec le modèle $MODEL"
    python $PYTHON_SCRIPT $MODEL 
done

cleanup

echo "Script terminé avec succès."