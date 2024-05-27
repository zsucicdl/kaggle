import os

from kaggle_scripts.comp_config import C_NAME
from setup.kaggle_api import KaggleApiBetter
from kaggle_scripts.utils.paths import get_competition_data_path

api = KaggleApiBetter()
competition_data_path = get_competition_data_path(C_NAME)

os.makedirs(C_NAME, exist_ok=True)
os.makedirs(os.path.join(C_NAME, "gold"), exist_ok=True)
os.makedirs(os.path.join(C_NAME, "silver"), exist_ok=True)
#os.makedirs(os.path.join(C_NAME, "bronze"), exist_ok=True)

#todu progressbar
print(C_NAME)

kernels = []
for i in range(1, 101):
    kernels_page = api.kernels_list(page=i, competition=C_NAME)
    kernels.extend(kernels_page)
    print(f"Page {i} done")

# Initialize the lists to store the notebooks in each category
gold_notebooks = []
silver_notebooks = []
bronze_notebooks = []

for notebook in kernels:
    if notebook.totalVotes >= 50:
        gold_notebooks.append(notebook)
    elif notebook.totalVotes >= 10:
        silver_notebooks.append(notebook)
    # elif notebook.totalVotes >= 5:
    #     bronze_notebooks.append(notebook)

# doing, done, check exists
# for notebook in gold_notebooks:
#     api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/gold')
# for notebook in silver_notebooks:
#     api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/silver')
# for notebook in bronze_notebooks:
#     api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/bronze')

# avoiding weird kaggle regex 500 error
# pull with try blocks and counting
gold_count = 0
for notebook in gold_notebooks:
    try:
        api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/gold')
        print(f"pulled {notebook.ref}")
        gold_count += 1
    except:
        pass
print(f"Gold notebooks pulled: {gold_count}")

silver_count = 0
for notebook in silver_notebooks:
    try:
        api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/silver')
        silver_count += 1
    except:
        pass
print(f"Silver notebooks pulled: {silver_count}")

# bronze_count = 0
# for notebook in bronze_notebooks:
#     try:
#         api.kernels_pull(kernel=notebook.ref, path=f'./{C_NAME}/bronze')
#         bronze_count += 1
#     except:
#         pass
# print(f"Bronze notebooks pulled: {bronze_count}")
#