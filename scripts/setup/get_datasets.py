import os
import zipfile

from scripts.setup.constants import datasets
from scripts.setup.kaggle_api import KaggleApiBetter
from scripts.utils.path_utils import get_competition_data_path

api = KaggleApiBetter()
competition_data_path = get_competition_data_path()

for kaggle_dataset_url in datasets:
    owner, dataset_name = kaggle_dataset_url.split('/')[-2:]
    kaggle_dataset_path=f'{owner}/{dataset_name}'
    api.dataset_download_files(dataset=kaggle_dataset_path, path=competition_data_path, force=False)
    local_dataset_path=f'{competition_data_path}/{dataset_name}'
    zip_file_name = f'{local_dataset_path}.zip'
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(local_dataset_path)
    os.remove(zip_file_name)
