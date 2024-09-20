import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


class KaggleApiBetter(KaggleApi):
    def __init__(self):
        self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})
