import re
from collections import Counter

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
from nltk.corpus import stopwords
# from autocorrect import Speller
from spellchecker import SpellChecker
from transformers import AutoTokenizer

from kaggle_scripts.utils.path_utils import get_competition_data_path


class DebertaPreprocessor:
    def __init__(self,
                 model_name: str,
                 ) -> None:
        competition_data_path = get_competition_data_path()
        self.tokenizer = AutoTokenizer.from_pretrained(f"{competition_data_path}/{model_name}")
        self.STOP_WORDS = set(stopwords.words('english'))

        self.spacy_ner_model = spacy.load('en_core_web_sm', )
        self.speller = SpellChecker()  # Speller(lang='en')

    def count_text_length(self, df: pd.DataFrame, col: str) -> pd.Series:
        """ text length """
        tokenizer = self.tokenizer
        return df[col].progress_apply(lambda x: len(tokenizer.encode(x)))

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """

        def check_is_stop_word(word):
            return word in self.STOP_WORDS

        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int):
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        # # Optionally, you can get the frequency of common n-grams for a more nuanced analysis
        # original_ngram_freq = Counter(ngrams(original_words, n))
        # summary_ngram_freq = Counter(ngrams(summary_words, n))
        # common_ngram_freq = {ngram: min(original_ngram_freq[ngram], summary_ngram_freq[ngram]) for ngram in common_ngrams}

        return len(common_ngrams)

    def ner_overlap_count(self, row, mode: str):
        model = self.spacy_ner_model

        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])

        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)

        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))

        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}

    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):

        wordlist = text.split()
        amount_miss = len(list(self.speller.unknown(wordlist)))

        return amount_miss

    def run(self,
            prompts: pd.DataFrame,
            summaries: pd.DataFrame,
            mode: str
            ) -> pd.DataFrame:

        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x),
                skip_special_tokens=True
            )
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x),
                skip_special_tokens=True
            )

        )
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(2,), axis=1
        )
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )

        # Crate dataframe with count of each category NERs overlap for all the summaries
        # Because it spends too much time for this feature, I don't use this time.
        #         ners_count_df  = input_df.progress_apply(
        #             lambda row: pd.Series(self.ner_overlap_count(row, mode=mode), dtype='float64'), axis=1
        #         ).fillna(0)
        #         self.ner_keys = ners_count_df.columns
        #         ners_count_df['sum'] = ners_count_df.sum(axis=1)
        #         ners_count_df.columns = ['NER_' + col for col in ners_count_df.columns]
        #         # join ner count dataframe with train dataframe
        #         input_df = pd.concat([input_df, ners_count_df], axis=1)

        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)

        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])
