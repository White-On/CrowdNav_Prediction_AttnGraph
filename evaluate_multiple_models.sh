#!/bin/bash

# We need to pass the model file as an argument
# The model file contains all the element of the model (weights, configuration, log) 
#  usually generated with the clean_model_empack.sh script
MODEL_FILE=$1

# Script Python à lancer
PYTHON_SCRIPT="main_PPO_stable_baseline.py"

# Fonction pour arrêter tous les processus enfants
cleanup() {
    echo "Arrêt des scripts..."
    pkill -P $$
    exit 1
}

# Capturer les signaux de terminaison
trap cleanup SIGINT SIGTERM

# Compter le nombre de fichiers de configuration
MODEL_VERSION=($MODEL_FILE/*)
NUM_VERSION=${#MODEL_VERSION[@]}

echo "Nombre de fichiers de configuration trouvés : $NUM_VERSION"
if [ $NUM_VERSION -eq 0 ]; then
    echo "Aucun fichier de configuration trouvé dans le dossier $MODEL_FILE"
    exit 1
fi

# Lancer un script Python pour chaque fichier de configuration
I=0
for MODEL_VERSION in "${MODEL_VERSION[@]}"; do
    echo "Lancement de $PYTHON_SCRIPT avec la configuration $MODEL_VERSION ($((I+1))/$NUM_VERSION)"
    I=$((I+1))
    # Dossier contenant les fichiers de configuration
    CONFIG_DIR=($MODEL_VERSION/*.gin)
    MODEL_NAME=($MODEL_VERSION/*.zip)
    python $PYTHON_SCRIPT -e -v --config $CONFIG_DIR --model $MODEL_NAME
done

echo "Tous les scripts ont été évalués."