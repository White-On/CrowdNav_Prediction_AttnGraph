#!/bin/bash

# Script pour ranger les modèles appris dans un dossier avec le nom du dernier commit Git
# avec leurs log tensorboard, fichiers de configuration associés et le model en lui meme

# Récupère la description du dernier commit Git
GIT_COMMIT_MESSAGE=$(git log -1 --pretty=%B)

# Dossier contenant les fichiers de configuration
CONFIG_DIRECTORY="gin_config_files"

# Nom de base du modèle et du fichier de log
BASE_MODEL_NAME="ppo_CrowdSimCar"
BASE_LOG_FILE_NAME="runs/PPO"

# Crée un fichier ayant pour nom la dernière description de commit Git en retirant les espaces
# et en remplaçant les caractères spéciaux par des tirets
MAIN_DIRECTORY=$(echo $GIT_COMMIT_MESSAGE | tr '[:space:]' '_' | tr '[:punct:]' '_')
MAIN_DIRECTORY="MODEL_FILE_${MAIN_DIRECTORY}"
mkdir -p $MAIN_DIRECTORY

# Compter le nombre de fichiers de configuration
CONFIG_FILES=($CONFIG_DIRECTORY/*.gin)
NUM_CONFIG_FILES=${#CONFIG_FILES[@]}

echo "Nombre de fichiers de configuration trouvés : $NUM_CONFIG_FILES"

# On crée un sous-dossier pour chaque fichier de configuration que l'on copie dedans
INDEX=0
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    INDEX=$((INDEX+1))
    SUBDIRECTORY="$MAIN_DIRECTORY/$INDEX"
    mkdir -p $SUBDIRECTORY
    cp $CONFIG_FILE $SUBDIRECTORY
    # echo "Copie du fichier de configuration $CONFIG_FILE dans $SUBDIRECTORY"
done

# On copie aussi le modèle et les logs tensorboard dans chaque sous-dossier 
for i in $(seq 1 $NUM_CONFIG_FILES); do
    MODEL_NAME="${BASE_MODEL_NAME}_${i}.zip"
    LOG_FILE_NAME="${BASE_LOG_FILE_NAME}_${i}_0"
    cp -r $MODEL_NAME $MAIN_DIRECTORY/$i
    cp -r $LOG_FILE_NAME $MAIN_DIRECTORY/$i
    # echo "Copie du modèle $MODEL_NAME et du fichier de log $LOG_FILE_NAME dans $MAIN_DIRECTORY/$i"
done

echo "Script terminé avec succès."