#!/bin/bash

# Script pour récupérer les modèles appris à partir de la machine distante
# et les copier localement ainsi que les logs tensorboard
# puis on range les fichiers dans un dossier avec le nom du dernier commit git 
# et enfin on supprime les fichiers inutiles


# Fonction pour fermer la connexion SSH persistante
cleanup() {
    echo "Fermeture de la connexion SSH persistante..."
    ssh -O exit -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST
    exit 0
}

# Capturer les signaux d'interruption et de terminaison
trap cleanup SIGINT SIGTERM

./retrive_model.sh

# Changez les valeurs pour correspondre à votre configuration .env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo ".env file not found"
    exit 1
fi

echo "Création d'une connexion SSH persistante à $REMOTE_USER@$REMOTE_HOST..."
ssh -M -f -N -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST
scp -r -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST:$LOG_FILE_PATH $LOCAL_RESULTS_PATH

./clean_model_empack.sh

rm -r runs/PPO*
rm -r ppo_CrowdSimCar*.zip

echo "Script terminé avec succès."

cleanup