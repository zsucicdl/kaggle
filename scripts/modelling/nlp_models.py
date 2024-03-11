import os

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer

from scripts.evaluation.metrics import compute_metrics
from scripts.utils.env_utils import seed_everything
from scripts.utils.path_utils import get_competition_data_path
from accelerate import Accelerator

# from autocorrect import Speller
# set random seed

class ContentScoreRegressor:
    def __init__(self,
                 model_name: str,
                 model_dir: str,
                 target: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 max_length: int,
                 ):
        self.inputs = ["prompt_text", "prompt_title", "prompt_question", "text"]
        self.input_col = "input"

        self.text_cols = [self.input_col]
        self.target = target
        self.target_cols = [target]

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length

        competition_data_path = get_competition_data_path()

        self.accelerator = Accelerator()
        self.device = self.accelerator.device


        self.tokenizer = AutoTokenizer.from_pretrained(f"{competition_data_path}/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"{competition_data_path}/{model_name}")

        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression",
        })

        seed_everything(seed=42)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        tokenized = self.tokenizer(examples[self.input_col],
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
        }

    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_length)
        return tokenized

    def train(self,
              fold: int,
              train_df: pd.DataFrame,
              valid_df: pd.DataFrame,
              batch_size: int,
              learning_rate: float,
              weight_decay: float,
              num_train_epochs: float,
              save_steps: int,
              ) -> None:
        """fine-tuning"""

        sep = self.tokenizer.sep_token
        train_df[self.input_col] = (
                train_df["prompt_title"] + sep
                + train_df["prompt_question"] + sep
                + train_df["text"]
        )

        valid_df[self.input_col] = (
                valid_df["prompt_title"] + sep
                + valid_df["prompt_question"] + sep
                + valid_df["text"]
        )

        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]

        competition_data_path = get_competition_data_path()
        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"{competition_data_path}/{self.model_name}",
            config=self.model_config
        )



        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False)

        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        # eg. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold))

        training_args = TrainingArguments(
            output_dir=model_fold_dir,
            load_best_model_at_end=True,  # select best model
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="rmse",
            save_total_limit=1
        )

        trainer = Trainer(
            model=model_content,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()

        model_content.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def predict(self,
                test_df: pd.DataFrame,
                fold: int,
                ):
        """predict content score"""

        sep = self.tokenizer.sep_token
        in_text = (
                test_df["prompt_title"] + sep
                + test_df["prompt_question"] + sep
                + test_df["text"]
        )
        test_df[self.input_col] = in_text

        test_ = test_df[[self.input_col]]

        test_dataset = Dataset.from_pandas(test_, preserve_index=False)
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)


        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        # Accelerate: Prepare your models
        model_content = self.accelerator.prepare(model_content)
        model_content.eval()
        # model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        # model_content.eval()

        # e.g. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold))

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=4,
            dataloader_drop_last=False,
        )

        # init trainer
        infer_content = Trainer(
            model=model_content,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=test_args)

        preds = infer_content.predict(test_tokenized_dataset)[0]

        return preds
