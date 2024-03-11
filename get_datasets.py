import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json


class KaggleApiBetter(KaggleApi):
    def __init__(self):
        # self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})
        with open("kaggle.json") as f:
            kaggle_json = json.load(f)
        self._load_config(kaggle_json)
        print(kaggle_json)
        # os.environ["KAGGLE_USERNAME"] = kaggle_json["username"]
        # os.environ["KAGGLE_KEY"] = kaggle_json["key"]
        # why interpreter does not see this?

        # hack
        # self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})


def get_data_directory_path(sentinel=".git"):
    """
    Returns the path to the data folder located at the project root.

    Parameters:
        - sentinel: The name of a file or folder that exists at the project root.
                   Defaults to ".git".

    Returns:
        Path object pointing to the data folder.
    """
    # Start from the current working directory
    current_path = Path.cwd()

    # Traverse upwards until the sentinel is found or root is reached
    while current_path != current_path.parent:
        if (current_path / sentinel).exists():
            return current_path / "input"
        current_path = current_path.parent

    # If we're here, the sentinel was not found. Handle appropriately (e.g., raise an exception).
    raise FileNotFoundError(
        f"Project root with sentinel '{sentinel}' not found from starting directory {Path.cwd()}"
    )


def get_data_path():
    data_path = get_data_directory_path()
    return data_path


api = KaggleApiBetter()
data_path = get_data_path()

datasets = [
    "https://www.kaggle.com/datasets/cdeotte/brain-spectrograms",
    "https://www.kaggle.com/datasets/cdeotte/brain-eeg-spectrograms",
    "https://www.kaggle.com/datasets/cdeotte/kaggle-kl-div",
    "https://www.kaggle.com/datasets/cdeotte/tf-efficientnet-imagenet-weights",
    "https://www.kaggle.com/datasets/cdeotte/brain-efficientnet-models-v3-v4-v5",
    "https://www.kaggle.com/datasets/cdeotte/brain-eegs"
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
