import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleApiBetter(KaggleApi):
    def __init__(self):
        #self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})
        with open('kaggle.json') as f:
            kaggle_json = json.load(f)
        self._load_config(kaggle_json)

        #os.environ["KAGGLE_USERNAME"] = kaggle_json["username"]
        #os.environ["KAGGLE_KEY"] = kaggle_json["key"]
        # why interpreter does not see this?

        # hack
        #self._load_config({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]})