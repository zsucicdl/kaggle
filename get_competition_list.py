from kaggle_api import KaggleApiBetter
from kaggle_scripts.utils.paths import get_competition_data_path

api = KaggleApiBetter()
competition_data_path = get_competition_data_path()

for c in api.competitions_list():
    print(str(c).split("competitions/")[1])
