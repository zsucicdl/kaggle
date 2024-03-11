from scripts.setup.kaggle_api import KaggleApiBetter

api = KaggleApiBetter()
print(api.competitions_list())
