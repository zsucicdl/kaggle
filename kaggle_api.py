import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import json
load_dotenv()


class KaggleApiBetter(KaggleApi):
    def __init__(self):
        with open("kaggle.json", "r") as f:
            kaggle_json = json.load(f)
        self._load_config(kaggle_json)
        #self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})