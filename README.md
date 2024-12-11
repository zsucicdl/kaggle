Work in Progress

1. GENERAL SETUP RECOMMENDATION
conda create --name kaggle_env python=3.8 -y
conda activate kaggle_env
conda install pip ipython ipykernel stdlib-list pipreqs kaggle -y
ipython kernel install --name kaggle_env --user
python -m pip install -r requirements.txt