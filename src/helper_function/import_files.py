import pandas as pd
from src.const import paths
from src.const import columns


def import_train_set(path=paths.FILE_TRAIN):
    df = pd.read_csv(path, index_col=columns.INDEX)
    return df


def import_clean_train_set(path=paths.FILE_TRAIN_CLEAR):
    df = pd.read_csv(path, index_col=columns.INDEX)
    return df
