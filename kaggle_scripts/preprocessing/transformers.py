from datasets import Dataset


# def get_dataset(df, columns_mapping=None):
#     data_dict = {new_col: df[old_col].tolist() for old_col, new_col in columns_mapping.items()}
#     ds = Dataset.from_dict(data_dict)
#     return ds
def get_transformers_dataset_from_pandas(df):
    ds = Dataset.from_pandas(df)
    return ds

class KaggleTokenizer(object):
    def __init__(self, tokenizer, max_length, truncation, batched):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.batched = batched

    def tokenize_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['full_text'], truncation=self.truncation, max_length=self.max_length
        )
        return tokenized_inputs

    def tokenize_dataset(self, dfs, columns_mappings=None):

        ds = get_dataset(df, columns_mapping)
        tokenized_ds = ds.map(self.tokenize_function, batched=True)
        tokenized_datasets.append(tokenized_ds)

        return tokenized_ds
