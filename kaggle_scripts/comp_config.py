from kaggle_scripts.utils.config import KaggleConfig
from kaggle_scripts.utils.paths import KagglePaths

C_NAME = 'um-game-playing-strength-of-mcts-variants'


class Aes2Config(KaggleConfig):
    def __init__(self, **kwargs):
        self.VER = kwargs.get('VER', '0001')
        self.LOAD_FROM = kwargs.get('LOAD_FROM', None)
        self.COMPUTE_CV = kwargs.get('COMPUTE_CV', True)
        self.n_splits = kwargs.get('n_splits', 5)
        self.seed = kwargs.get('seed', 42)
        self.max_length = kwargs.get('max_length', 1024)
        self.lr = kwargs.get('lr', 1e-5)
        self.train_batch_size = kwargs.get('train_batch_size', 2)
        self.eval_batch_size = kwargs.get('eval_batch_size', 8)
        self.train_epochs = kwargs.get('train_epochs', 4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.warmup_ratio = kwargs.get('warmup_ratio', 0.0)
        self.num_labels = kwargs.get('num_labels', 6)
        self.folding_strategy = kwargs.get('folding_strategy', 'StratifiedKFold')
        self.tokenization_strategy = kwargs.get('tokenization_strategy', 'Aes2TokenizerV1')
        self.tokenization_column = kwargs.get('tokenization_column', 'full_text')
        super().__init__(self.seed)


class Aes2Paths(KagglePaths):
    def __init__(self):
        super().__init__(C_NAME)
        self.train_valid_path = self.get_competition_data_path('train.csv')
        self.test_path = self.get_competition_data_path('test.csv')
        self.sub_path = self.get_competition_data_path('sample_submission.csv')
        self.model_path = self.get_dataset_data_path('huggingfacedebertav3variants/deberta-v3-large')
        self.model_path = 'deepset/deberta-v3-large-squad2'

PATHS = Aes2Paths()
CFG = Aes2Config()