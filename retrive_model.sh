#!/bin/bash

# Permet d'aller chercher les modèles appris sur la machine distante

# Changez les valeurs pour correspondre à votre configuration .env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo ".env file not found"
    exit 1
fi

# Fonction pour fermer la connexion SSH persistante
cleanup() {
    echo "Fermeture de la connexion SSH persistante..."
    ssh -O exit -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST
    exit 0
}

# Créez une connexion SSH persistante à la machine distante
echo "Création d'une connexion SSH persistante à $REMOTE_USER@$REMOTE_HOST..."
ssh -M -f -N -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST

# # Définir les modèles à copier
# models=("ppo_CrowdSimCar.zip" "ddpg_CrowdSimCar.zip" "recurrent_ppo_CrowdSimCar.zip")

# On définis les modèle a récupéré c'est a dire tout les fichiers avec une extention .zip
models=($(ssh -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST "ls $REMOTE_MODEL_PATH/*.zip"))

# Copier chaque modèle
for model in "${models[@]}"; do
  echo "Copie du modèle appris à partir de $REMOTE_MODEL_PATH vers $LOCAL_MODEL_PATH..."
  scp -r -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST:$model ./
done

# Fermez la connexion SSH persistante
cleanup