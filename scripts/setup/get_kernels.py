import os

from scripts.setup.constants import C_NAME
from scripts.setup.kaggle_api import KaggleApiBetter
# wrong  directory
api = KaggleApiBetter()

print(api.competitions_list())

os.makedirs(C_NAME, exist_ok=True)
os.makedirs(os.path.join(C_NAME, "gold"), exist_ok=True)
os.makedirs(os.path.join(C_NAME, "silver"), exist_ok=True)
os.makedirs(os.path.join(C_NAME, "bronze"), exist_ok=True)

kernels = []
for i in range(1, 101):
    kernels_page = api.kernels_list(page=i, competition=C_NAME)
    kernels.extend(kernels_page)

gold_notebooks = []
silver_notebooks = []
bronze_notebooks = []

for notebook in kernels:
    if notebook.totalVotes >= 50:
        gold_notebooks.append(notebook)
    elif notebook.totalVotes >= 10:
        silver_notebooks.append(notebook)
    elif notebook.totalVotes >= 5:
        bronze_notebooks.append(notebook)

for notebook in gold_notebooks:
    api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/gold')
for notebook in silver_notebooks:
    api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/silver')
for notebook in bronze_notebooks:
    api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/bronze')
