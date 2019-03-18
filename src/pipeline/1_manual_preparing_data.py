import pandas as pd

from src.const import features
from src.const import paths
from src.helper_function import import_files


def manual_preprocessing(save=True):
    """
    Preapare data for training. Output is a a DataFrame with numerical features.
    :param save: Bool, option to save data.
    :return: DataFrame
    """
    df = import_files.import_all_data()

    # Convert categorical variable into dummy/indicator variables
    df = pd.get_dummies(df, columns=features.CAT_FEATURES)

    # Convert date to correct format
    df[features.DATE] = pd.to_datetime(df[features.DATE].astype(str).str[0:8], format='%Y%m%dT')

    # Create columns with months and days
    df['date_month'] = df[features.DATE].dt.month
    df['date_day'] = df[features.DATE].dt.day

    # Remove column "date"
    df.drop([features.DATE], axis=1, inplace=True)

    # Save dataFrame
    if save:
        df.to_csv(paths.FILE_TRAIN_CLEAR)
    return df


if __name__ == '__main__':
    manual_preprocessing()
