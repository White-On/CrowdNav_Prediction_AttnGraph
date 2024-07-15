#!/bin/bash

# Dossier contenant les fichiers de configuration
CONFIG_DIR="gin_config_files"

# Script Python à lancer
PYTHON_SCRIPT="main_PPO_stable_baseline.py"

# Nom du model et du fichier de log de base
BASE_MODEL_NAME="ppo_CrowdSimCar"
BASE_LOG_FILE="PPO"


# Fonction pour arrêter tous les processus enfants
cleanup() {
    echo "Arrêt des scripts..."
    pkill -P $$
    exit 1
}

# Capturer les signaux de terminaison
trap cleanup SIGINT SIGTERM

# Compter le nombre de fichiers de configuration
CONFIG_FILES=($CONFIG_DIR/*.gin)
NUM_CONFIGS=${#CONFIG_FILES[@]}

echo "Nombre de fichiers de configuration trouvés : $NUM_CONFIGS"
if [ $NUM_CONFIGS -eq 0 ]; then
    echo "Aucun fichier de configuration trouvé dans le dossier $CONFIG_DIR"
    exit 1
fi

# Lancer un script Python pour chaque fichier de configuration
I=0
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Lancement de $PYTHON_SCRIPT avec la configuration $CONFIG_FILE ($((I+1))/$NUM_CONFIGS)"
    I=$((I+1))
    MODEL_NAME="${BASE_MODEL_NAME}_${I}"
    python $PYTHON_SCRIPT -e -v --config $CONFIG_FILE --model $MODEL_NAME
done

# Attendre que tous les scripts Python se terminent
wait

echo "Tous les scripts ont été évalués."