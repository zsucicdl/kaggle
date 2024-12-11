from transformers import AutoTokenizer, AddedToken


class KaggleTokenizer(object):
    def __init__(self, model_path, max_length, truncation, batched, tokenization_column):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.truncation = truncation
        self.batched = batched
        self.tokenization_column = tokenization_column

    def tokenize_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.tokenization_column],
            truncation=self.truncation,
            max_length=self.max_length
        )
        return tokenized_inputs

    def tokenize_dataset(self, ds):
        return ds.map(self.tokenize_function, batched=self.batched)


class Aes2TokenizerV1(KaggleTokenizer):
    def __init__(self, model_path, max_length, truncation, batched, tokenization_column):
        super().__init__(model_path, max_length, truncation, batched, tokenization_column)
        self.tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        self.tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])


def get_tokenizer(tokenization_strategy, model_path, max_length, truncation, batched, tokenization_column):
    if tokenization_strategy == "Aes2TokenizerV1":
        return Aes2TokenizerV1(model_path, max_length, truncation, batched, tokenization_column)
    else:
        raise ValueError(f"Unknown tokenization_strategy: {tokenization_strategy}")
