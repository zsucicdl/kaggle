import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json

from setup.kaggle_api import KaggleApiBetter
from kaggle_scripts.utils.paths import get_data_path

api = KaggleApiBetter()
data_path = get_data_path()

datasets = [
    "https://www.kaggle.com/datasets/verracodeguacas/huggingfacedebertav3variants",
    "https://www.kaggle.com/datasets/cdeotte/deberta-v3-small-finetuned-v1"
]
#"https://www.kaggle.com/datasets/cdeotte/tf-efficientnet-whl-files",

for kaggle_dataset_url in datasets:
    owner, dataset_name = kaggle_dataset_url.split("/")[-2:]
    kaggle_dataset_path = f"{owner}/{dataset_name}"
    # check if dataset already exists

    local_dataset_path = f"{data_path}/{dataset_name}"
    if os.path.exists(local_dataset_path):
        print(f"{local_dataset_path} already exists, skipping download")
        continue
    print(f"{local_dataset_path} a")
    print(f"{local_dataset_path} bb")

    api.dataset_download_files(dataset=kaggle_dataset_path, path=data_path, force=False)
    zip_file_name = f"{local_dataset_path}.zip"
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        zip_ref.extractall(local_dataset_path)
    os.remove(zip_file_name)
