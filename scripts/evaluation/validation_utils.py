import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
import polars as pl
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
from scripts.preprocessing.nlp import *

import warnings

# TODO
def validate(
        train_df: pd.DataFrame,
        target: str,
        save_each_model: bool,
        model_name: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_length: int,
        n_splits: int,
        model
) -> pd.DataFrame:
    """predict oof data"""
    for fold in range(n_splits):
        print(f"fold {fold}:")

        valid_data = train_df[train_df["fold"] == fold]

        if save_each_model == True:
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else:
            model_dir = f"{model_name}/fold_{fold}"

        # csr = ContentScoreRegressor(
        #     model_name=model_name,
        #     target=target,
        #     model_dir=model_dir,  # モデル・foldごとにモデルファイルの保存先のdirを分ける
        #     hidden_dropout_prob=hidden_dropout_prob,
        #     attention_probs_dropout_prob=attention_probs_dropout_prob,
        #     max_length=max_length,
        # )

        pred = model.predict(
            test_df=valid_data,
            fold=fold
        )

        train_df.loc[valid_data.index, f"{target}_pred"] = pred

    return train_df


def split_group_k_fold(df, group_column_name, fold_column_name='fold', n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    for i, (_, val_index) in enumerate(gkf.split(df, groups=df[group_column_name])):
        df.loc[val_index, fold_column_name] = i
    return df


def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid


def evaluate(data_x, data_y, model, random_state=42, n_splits=5, test_x=None):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    test_y = np.zeros(len(data_x)) if (test_x is None) else np.zeros((len(test_x), n_splits))
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y.astype(str))):
        train_x, train_y, valid_x, valid_y = train_valid_split(data_x, data_y, train_index, valid_index)
        model.fit(train_x, train_y)
        if test_x is None:
            test_y[valid_index] = model.predict(valid_x)
        else:
            test_y[:, i] = model.predict(test_x)
    return test_y if (test_x is None) else np.mean(test_y, axis=1)



