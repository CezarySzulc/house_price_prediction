import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

from src.const import paths
from src.const import features
from src.helper_function import import_files
from src.helper_function import build_xgboost


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def auto_preprocessing(number_iteration_to_gen_features=10, generate_until_improved=True, save=True):
    """
    Generate more features. USed xgb model for this.
    :param number_iteration_to_gen_features: Number of iteration for generate more features
    :param generate_until_improved: Bool if 'True' then generating as many features as score will be improving
    :param save: Bool, option to save data.
    :return: DataFrames (train and test) with all features
    """
    df = import_files.import_clean_data()

    # Split to train and test set
    df_target = df[features.PRICE]
    df_data = df.drop(features.PRICE, axis=1)
    df_train_data, df_test_data, df_train_target, df_test_target = train_test_split(
        df_data, df_target, test_size=0.2, random_state=42)

    # Generate more features in loop
    if generate_until_improved:
        scoring_improvment = 1
        while scoring_improvment:
            df_train_data, df_test_data, scoring_improvment = build_xgboost.generate_more_feature_by_xgb(
                df_train_data, df_train_target, df_test_data, df_test_target, calculate_score_with_new_features=True
            )
            scoring_improvment = scoring_improvment > 0
    else:
        for _ in range(number_iteration_to_gen_features):
            df_train_data, df_test_data = build_xgboost.generate_more_feature_by_xgb(
                df_train_data, df_train_target, df_test_data, df_test_target, calculate_score_with_new_features=False
            )

    df_train = pd.merge(df_train_data, df_train_target, left_index=True, right_index=True)
    df_test = pd.merge(df_test_data, df_test_target, left_index=True, right_index=True)

    # Save data
    if save:
        df_train.to_csv(paths.FILE_TRAIN_AUTO)
        df_test.to_csv(paths.FILE_TEST_AUTO)

    return df_train, df_test


if __name__ == '__main__':
    auto_preprocessing()
