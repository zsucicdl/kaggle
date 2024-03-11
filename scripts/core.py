from abc import ABC, abstractmethod


class DataHandler(ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def load_data(self):
        pass


class Preprocessor(ABC):
    def __init__(self, strategies):
        self.strategies = strategies

    @abstractmethod
    def preprocess(self, df):
        pass


class FeatureEngineer(ABC):
    def __init__(self, features):
        self.features = features

    @abstractmethod
    def engineer_features(self, df):
        pass


class ModelTrainer(ABC):
    def __init__(self, model, params):
        self.model = model
        self.params = params

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class Validator(ABC):
    def __init__(self, metric):
        self.metric = metric

    @abstractmethod
    def validate(self, y_true, y_pred):
        pass


class Logger(ABC):
    def __init__(self, logger_type):
        self.logger_type = logger_type

    @abstractmethod
    def log(self, params, metrics):
        pass


# Factory patterns
class DataHandlerFactory:
    @staticmethod
    def create_data_handler(file_path):
        # Return a specific implementation of DataHandler
        pass


class PreprocessorFactory:
    @staticmethod
    def create_preprocessor(strategies):
        # Return a specific implementation of Preprocessor
        pass


class FeatureEngineerFactory:
    @staticmethod
    def create_feature_engineer(features):
        # Return a specific implementation of FeatureEngineer
        pass


class ModelTrainerFactory:
    @staticmethod
    def create_model_trainer(model, params):
        # Return a specific implementation of ModelTrainer
        pass


class ValidatorFactory:
    @staticmethod
    def create_validator(metric):
        # Return a specific implementation of Validator
        pass


class LoggerFactory:
    @staticmethod
    def create_logger(logger_type):
        # Return a specific implementation of Logger
        pass
