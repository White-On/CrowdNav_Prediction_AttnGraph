#!/bin/bash

# lance l'entrainement de plusieurs modèles en parallèle
# en utilisant des fichiers de configuration GIN différents

# Dossier contenant les fichiers de configuration
CONFIG_DIR="gin_config_files"

# Script Python à lancer
PYTHON_SCRIPT="main_PPO_stable_baseline.py"

# Nom du model et du fichier de log de base
BASE_MODEL_NAME="ppo_CrowdSimCar"
BASE_LOG_FILE="PPO"

# Nombre maximum de scripts Python à exécuter simultanément
MAX_CONCURRENT_JOBS=15

# Fonction pour arrêter tous les processus enfants
cleanup() {
    echo "Arrêt des scripts Python..."
    pkill -P $$
    exit 1
}
# Verifier si on exécute le script avec l'envirement virtuel de python .venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Vous devez activer l'environnement virtuel avant d'exécuter ce script."
fi

# Capturer les signaux de terminaison
trap cleanup SIGINT SIGTERM

# Générer un fichier de configuration GIN pour chaque fichier de configuration avec le template
python generate_gin_config_file.py

# Compter le nombre de fichiers de configuration
CONFIG_FILES=($CONFIG_DIR/*.gin)
NUM_CONFIGS=${#CONFIG_FILES[@]}

echo "Nombre de fichiers de configuration trouvés : $NUM_CONFIGS"
if [ $NUM_CONFIGS -eq 0 ]; then
    echo "Aucun fichier de configuration trouvé dans le dossier $CONFIG_DIR"
    exit 1
fi

if [ $NUM_CONFIGS -gt $MAX_CONCURRENT_JOBS ]; then
    echo "Nombre de fichiers de configuration ($NUM_CONFIGS) supérieur au nombre maximum de scripts Python à exécuter simultanément ($MAX_CONCURRENT_JOBS)"
    exit 1
fi

# Lancer un script Python pour chaque fichier de configuration
I=0
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Lancement de $PYTHON_SCRIPT avec la configuration $CONFIG_FILE ($((I+1))/$NUM_CONFIGS)"
    I=$((I+1))
    MODEL_NAME="${BASE_MODEL_NAME}_${I}"
    LOG_FILE="${BASE_LOG_FILE}_${I}"
    python $PYTHON_SCRIPT --config $CONFIG_FILE --model $MODEL_NAME --log $LOG_FILE > /dev/null 2>&1 &
done

# Attendre que tous les scripts Python se terminent
wait

echo "Tous les scripts de formation sont terminés."