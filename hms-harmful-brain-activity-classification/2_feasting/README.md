conda create --name kaggle_env python=3.8 -y
conda activate kaggle_env
conda install pip ipython ipykernel stdlib-list pipreqs kaggle -y
# pip install ipython ipykernel stdlib-list pipreqs kaggle
ipython kernel install --name kaggle_env --user
python -m pip install -r requirements.txt


REPRODUCING - until working
replace '/kaggle with '/home/zsucic/kaggle
seed
localreqs

PIPELINING - until working
feast
code structure
same metrics
replace /home/zsucic/kaggle with universal
generally enable submit notebook
black reformat



SECONDARY MODELS
vit
chrononet
xgboost
cnn
multimodal