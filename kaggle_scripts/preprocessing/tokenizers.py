from datasets import Dataset


class Tokenize(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_dataset(self, df, columns_mapping=None):
        """
        Converts a DataFrame to a Dataset. Optionally maps DataFrame columns to desired Dataset fields.

        Parameters:
        - df (pd.DataFrame): The DataFrame to convert.
        - columns_mapping (dict): Optional dictionary to map DataFrame columns to Dataset fields.
                                  Example: {'essay_id': 'id', 'full_text': 'text', 'label': 'target'}

        Returns:
        - ds (Dataset): The Hugging Face Dataset.
        """
        if columns_mapping is None:
            columns_mapping = {'essay_id': 'essay_id', 'full_text': 'full_text', 'label': 'label'}

        data_dict = {new_col: df[old_col].tolist() for old_col, new_col in columns_mapping.items()}
        ds = Dataset.from_dict(data_dict)
        return ds

    def tokenize_function(self, examples):
        """
        Tokenizes text examples.

        Parameters:
        - examples (dict): A dictionary containing text examples to tokenize.

        Returns:
        - tokenized_inputs (dict): A dictionary containing tokenized inputs.
        """
        tokenized_inputs = self.tokenizer(
            examples['full_text'], truncation=True, max_length=self.max_length
        )
        return tokenized_inputs

    def tokenize_datasets(self, dfs, columns_mappings=None):
        """
        Tokenizes multiple datasets.

        Parameters:
        - dfs (list): A list of DataFrames to tokenize.
        - columns_mappings (list): Optional list of dictionaries for columns mapping for each DataFrame.

        Returns:
        - tokenized_datasets (list): A list of tokenized datasets.
        """
        if columns_mappings is None:
            columns_mappings = [None] * len(dfs)

        tokenized_datasets = []
        for df, columns_mapping in zip(dfs, columns_mappings):
            ds = self.get_dataset(df, columns_mapping)
            tokenized_ds = ds.map(self.tokenize_function, batched=True)
            tokenized_datasets.append(tokenized_ds)

        return tokenized_datasets

# Example usage:
# tokenizer = YourTokenizer()
# max_length = 512
# tokenize_instance = Tokenize(tokenizer, max_length)
# tokenized_train, tokenized_valid = tokenize_instance.tokenize_datasets([train_df, valid_df])
