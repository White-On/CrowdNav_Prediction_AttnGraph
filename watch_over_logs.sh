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

# Fonction pour fermer la connexion SSH persistante
cleanup() {
    echo "Fermeture de la connexion SSH persistante..."
    ssh -O exit -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST
    exit 0
}

# Capturer les signaux d'interruption et de terminaison
trap cleanup SIGINT SIGTERM

# Utilisez scp pour copier le fichier de résultats à votre machine locale toutes les 30 secondes
echo "Début de la copie du fichier de résultats $LOG_FILE_PATH à l'emplacement local $LOCAL_RESULTS_PATH toutes les 30 secondes..."
while true;
do
  scp -r -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST:$LOG_FILE_PATH $LOCAL_RESULTS_PATH
  echo "Fichier de résultats copié, prochaine copie dans 30 secondes..."
  sleep 30
done
