import pandas as pd
from src.const import paths
from src.const import features


def import_all_data(path=paths.FILE_TRAIN):
    df = pd.read_csv(path, index_col=features.INDEX)
    return df


def import_clean_data(path=paths.FILE_TRAIN_CLEAR):
    df = pd.read_csv(path, index_col=features.INDEX)
    return df


def import_auto_train_data(path=paths.FILE_TRAIN_AUTO):
    df = pd.read_csv(path, index_col=features.INDEX)
    return df


def import_auto_test_data(path=paths.FILE_TEST_AUTO):
    df = pd.read_csv(path, index_col=features.INDEX)
    return df
