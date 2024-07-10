#!/bin/bash

# Changez les valeurs pour correspondre à votre configuration .env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo ".env file not found"
    exit 1
fi

# Si le premier argument est PPO ou DDPG, utilisez ce modèle
# on prendra PPO par défaut
if [ "$1" == "PPO" ]; then
    PYTHON_SCRIPT_PATH=$PPO_SCRIPT_PATH
    echo "PPO"
elif [ "$1" == "DDPG" ]; then
    PYTHON_SCRIPT_PATH=$DDPG_SCRIPT_PATH
    echo "DDPG"
else
    PYTHON_SCRIPT_PATH=$PPO_SCRIPT_PATH
    echo "default PPO"
fi


# Créez une connexion SSH persistante à la machine distante
echo "Création d'une connexion SSH persistante à $REMOTE_USER@$REMOTE_HOST..."
ssh -M -f -N -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST

# Exécutez votre programme Python dans un nouveau shell interactif qui active l'environnement virtuel .venv
echo "Exécution du script Python $PYTHON_SCRIPT_PATH dans un nouvel environnement virtuel..."

PYTHON_PID=$(ssh -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST "source $VENV_PATH/bin/activate; nohup python $PYTHON_SCRIPT_PATH > $OUTPUT_FILE 2>&1 & echo \$!")
echo "Le script Python est en cours d'exécution avec le PID $PYTHON_PID..."

# Définir un piège pour tuer le processus Python lorsque ce script est interrompu
trap "ssh -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST kill $PYTHON_PID" INT

# Utilisez scp pour copier le fichier de résultats à votre machine locale toutes les 30 secondes
echo "Début de la copie du fichier de résultats $RESULTS_FILE_PATH à l'emplacement local $LOCAL_RESULTS_PATH toutes les 30 secondes..."
while ssh -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST kill -0 $PYTHON_PID 2>/dev/null
do
  scp -r -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST:$RESULTS_FILE_PATH $LOCAL_RESULTS_PATH
  echo "Fichier de résultats copié, prochaine copie dans 30 secondes..."
  sleep 30
done

# Fermez la connexion SSH persistante
echo "Fermeture de la connexion SSH persistante..."
ssh -O exit -o ControlPath=$SSH_CONTROL_PATH $REMOTE_USER@$REMOTE_HOST
