import logging
import os
import random
import warnings

import numpy as np
import torch
# from autocorrect import Speller
from datasets import disable_progress_bar
from tqdm import tqdm


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_env():
    warnings.simplefilter("ignore")
    logging.disable(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    disable_progress_bar()
    tqdm.pandas()
