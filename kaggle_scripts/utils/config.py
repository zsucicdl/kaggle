from kaggle_scripts.utils.envs import init_env, seed_everything


class KaggleConfig():
    def __init__(self, seed):
        init_env()
        seed_everything(seed)
