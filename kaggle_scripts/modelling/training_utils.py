import os
import shutil

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# from autocorrect import Speller
# set random seed
from kaggle_scripts.modelling.model_configs import DebertaConfig
from kaggle_scripts.modelling.nlp_models import ContentScoreRegressor


def train_by_fold(
        train_df: pd.DataFrame,
        model_name: str,
        target: str,
        save_each_model: bool,
        n_splits: int,
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length: int
):
    # delete old model files
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    os.mkdir(model_name)

    for fold in range(DebertaConfig.n_splits):
        print(f"fold {fold}:")

        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]

        if save_each_model == True:
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else:
            model_dir = f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )

        csr.train(
            fold=fold,
            train_df=train_data,
            valid_df=valid_data,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )


def predict(
        test_df: pd.DataFrame,
        target: str,
        save_each_model: bool,
        model_name: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_length: int
):
    """predict using mean folds"""

    for fold in range(DebertaConfig.n_splits):
        print(f"fold {fold}:")

        if save_each_model == True:
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else:
            model_dir = f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )

        pred = csr.predict(
            test_df=test_df,
            fold=fold
        )

        test_df[f"{target}_pred_{fold}"] = pred

    test_df[f"{target}"] = test_df[[f"{target}_pred_{fold}" for fold in range(DebertaConfig.n_splits)]].mean(axis=1)

    return test_df
