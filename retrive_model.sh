#!/bin/bash

# Changez les valeurs pour correspondre à votre configuration .env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo ".env file not found"
    exit 1
fi

# Créez une connexion SSH persistante à la machine distante
echo "Création d'une connexion SSH persistante à $REMOTE_USER@$REMOTE_HOST..."
ssh -M -f -N -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST

# Définir les modèles à copier
models=("ppo_CrowdSimCar.zip" "ddpg_CrowdSimCar.zip" "recurrent_ppo_CrowdSimCar.zip")

# Copier chaque modèle
for model in "${models[@]}"; do
  echo "Copie du modèle appris à partir de $REMOTE_MODEL_PATH vers $LOCAL_MODEL_PATH..."
  scp -r -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST:$model ./
done
