import os

# export LC_ALL=C
os.environ['LC_ALL'] = 'C'
import zipfile
import os

from kaggle_api import KaggleApiBetter
from kaggle_scripts.utils.paths import get_competition_data_path
C_NAME = 'um-game-playing-strength-of-mcts-variants'
api = KaggleApiBetter()
competition_data_path = get_competition_data_path(C_NAME)
os.makedirs(competition_data_path, exist_ok=True)

print(0)
# from datetime import datetime
# date_str = '11 02 2024 20:45:41'
# date_obj = datetime.strptime(date_str, '%d %b %Y %H:%M:%S')
# print(str(date_obj))
# from datetime import datetime
# date_str = 'Wed, 10 Jan 2024 20:45:41 GMT'
# date_obj = datetime.strptime(date_str, '%a, %d %m %Y %H:%M:%S %Z')
import locale

print(locale.getlocale())

# date_str = 'Wed, 10 Jan 2024 20:45:41'
# date_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S')
# print(str(date_obj))

# from dateutil.parser import parse
#
# date_str = 'Wed, 10 Jan 2024 20:45:41 GMT'
# date_obj = parse(date_str)


print(f"Downloading data to {competition_data_path}")
api.competition_download_files(competition=C_NAME, path=competition_data_path, force=True)

zip_file_name = os.path.join(competition_data_path, f"{C_NAME}.zip")
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(competition_data_path)
os.remove(zip_file_name)
