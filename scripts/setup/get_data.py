import os
import zipfile

from scripts.setup.constants import C_NAME
from scripts.setup.kaggle_api import KaggleApiBetter
from scripts.utils.path_utils import get_competition_data_path

api = KaggleApiBetter()
competition_data_path = get_competition_data_path()
os.makedirs(competition_data_path, exist_ok=True)
api.competition_download_files(competition=C_NAME, path=competition_data_path, force=True)

zip_file_name = os.path.join(competition_data_path, f"{C_NAME}.zip")
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(competition_data_path)
os.remove(zip_file_name)
