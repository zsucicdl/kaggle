import gc
from math import pi, sqrt, exp

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
from torch.utils.data import Dataset
from tqdm import tqdm

from kaggle_scripts.utils.paths import get_competition_data_path


def simple_load_commonlit_all_data(rows=None):
    competition_data_path = get_competition_data_path()

    prompts_train = pd.read_csv(competition_data_path / "prompts_train.csv", nrows=rows)
    prompts_test = pd.read_csv(competition_data_path / "prompts_test.csv", nrows=rows)
    summaries_train = pd.read_csv(competition_data_path / "summaries_train.csv", nrows=rows)
    summaries_test = pd.read_csv(competition_data_path / "summaries_test.csv", nrows=rows)
    sample_submission = pd.read_csv(competition_data_path / "sample_submission.csv", nrows=rows)
    return prompts_train, prompts_test, summaries_train, summaries_test, sample_submission


class DataReader:
    def __init__(self, demo_mode, main_dir):
        super().__init__()
        submission_path = main_dir + "sample_submission.csv"
        train_events = main_dir + "train_events.csv"
        # PARQUET FILES:
        train_series = main_dir + "train_series.parquet"
        test_series = main_dir + "test_series.parquet"

        self.names_mapping = {
            "submission": {"path": submission_path, "is_parquet": False, "has_timestamp": False},
            "train_events": {"path": train_events, "is_parquet": False, "has_timestamp": True},
            "train_series": {"path": train_series, "is_parquet": True, "has_timestamp": True},
            "test_series": {"path": test_series, "is_parquet": True, "has_timestamp": True}
        }
        self.valid_names = ["submission", "train_events", "train_series", "test_series"]
        self.demo_mode = demo_mode

    def verify(self, data_name):
        "function for data name verification"
        if data_name not in self.valid_names:
            print("PLEASE ENTER A VALID DATASET NAME, VALID NAMES ARE : ", self.valid_names)
        return

    def cleaning(self, data):
        "cleaning function : drop na values"
        before_cleaning = len(data)
        print("Number of missing timestamps : ", len(data[data["timestamp"].isna()]))
        data = data.dropna(subset=["timestamp"])
        after_cleaning = len(data)
        print("Percentage of removed rows : {:.1f}%".format(100 * (before_cleaning - after_cleaning) / before_cleaning))
        #         print(data.isna().any())
        #         data = data.bfill()
        return data

    @staticmethod
    def reduce_memory_usage(data):
        "iterate through all the columns of a dataframe and modify the data type to reduce memory usage."
        start_mem = data.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        for col in data.columns:
            col_type = data[col].dtype
            if col_type != object:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
            else:
                data[col] = data[col].astype('category')

        end_mem = data.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return data

    def load_data(self, data_name):
        "function for data loading"
        self.verify(data_name)
        data_props = self.names_mapping[data_name]
        if data_props["is_parquet"]:
            if self.demo_mode:
                pf = ParquetFile(data_props["path"])
                demo_rows = next(pf.iter_batches(batch_size=20_000))
                data = pa.Table.from_batches([demo_rows]).to_pandas()
            else:
                data = pd.read_parquet(data_props["path"])
        else:
            if self.demo_mode:
                data = pd.read_csv(data_props["path"], nrows=20_000)
            else:
                data = pd.read_csv(data_props["path"])

        gc.collect()
        if data_props["has_timestamp"]:
            print('cleaning')
            data = self.cleaning(data)
            gc.collect()
        # data = self.reduce_memory_usage(data)
        return data


class SleepTestDataset(Dataset):
    def __init__(
            self,
            test_series,
            test_ids,
            sigma
    ):
        self.enmo_mean = np.load('/kaggle/input/detect-sleep-states-dataprepare/enmo_mean.npy')
        self.enmo_std = np.load('/kaggle/input/detect-sleep-states-dataprepare/enmo_std.npy')

        self.Xs = self.conv_dfs(test_series, test_ids)

        self.feat_list = np.load('/kaggle/input/detect-sleep-states-train/feature_list.npy')
        self.label_list = ['onset', 'wakeup']

        self.hour_feat = ['hour']

        self.sigma=sigma

    def conv_dfs(self, series, ids):
        res = []
        for j, viz_id in tqdm(enumerate(ids), total=len(ids)):
            viz_series = series.loc[(series.series_id == viz_id)].copy().reset_index()
            viz_series['dt'] = pd.to_datetime(viz_series.timestamp, format='%Y-%m-%dT%H:%M:%S%z').astype(
                "datetime64[ns, UTC-04:00]")
            viz_series['hour'] = viz_series['dt'].dt.hour
            new_df = viz_series[['step', 'anglez', 'enmo', 'hour']]
            res.append(new_df)

        return res

    def norm_feat_eng(self, X, init=False):
        X['anglez'] = X['anglez'] / 90.0
        X['enmo'] = (X['enmo'] - self.enmo_mean) / (self.enmo_std + 1e-8)

        for w in [1, 2, 4, 8, 16]:
            X['anglez_shift_pos_' + str(w)] = X['anglez'].shift(w).fillna(0)
            X['anglez_shift_neg_' + str(w)] = X['anglez'].shift(-w).fillna(0)

            X['enmo_shift_pos_' + str(w)] = X['enmo'].shift(w).fillna(0)
            X['enmo_shift_neg_' + str(w)] = X['enmo'].shift(-w).fillna(0)

            if init:
                self.feat_list.append('anglez_shift_pos_' + str(w))
                self.feat_list.append('anglez_shift_neg_' + str(w))

                self.feat_list.append('enmo_shift_pos_' + str(w))
                self.feat_list.append('enmo_shift_neg_' + str(w))

        for r in [17, 33, 65]:
            tmp_anglez = X['anglez'].rolling(r, center=True)
            X[f'anglez_mean_{r}'] = tmp_anglez.mean()
            X[f'anglez_std_{r}'] = tmp_anglez.std()

            tmp_enmo = X['enmo'].rolling(r, center=True)
            X[f'enmo_mean_{r}'] = tmp_enmo.mean()
            X[f'enmo_std_{r}'] = tmp_enmo.std()

            if init:
                self.feat_list.append(f'anglez_mean_{r}')
                self.feat_list.append(f'anglez_std_{r}')

                self.feat_list.append(f'enmo_mean_{r}')
                self.feat_list.append(f'enmo_std_{r}')

        X = X.fillna(0)

        return X.astype(np.float32)

    def gauss(self, n, sigma):
        # guassian distribution function
        if n is None:
            n= self.sigma
        if sigma is None:
            sigma = self.sigma * 0.15


        r = range(-int(n / 2), int(n / 2) + 1)
        return [1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index):
        X = self.Xs[index]
        X = self.norm_feat_eng(X, init=False)
        x = X[self.feat_list].values.astype(np.float32)
        t = X[self.hour_feat].values.astype(np.int32)
        return x, t
