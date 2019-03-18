from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def generate_more_feature_by_xgb(
        df_data,
        df_target,
        df_data_test,
        df_target_test,
        calculate_score_with_new_features=True):
    """
    Generate more features, first build model using XGBoost method.
    Select most important features and generate extra features.
    Extra features are sum and multiplication of 2 most important features and
    log for best features.
    :param df_data: DataFrame with data to train model
    :param df_target: DataFrame with correct output for the model
    :param df_data_test: DataFrame with data to test model
    :param df_target_test: DataFrame with test output for the model
    :param calculate_score_with_new_features: Bool, retrain model again and calculate accuracy score
    :return: DataFrames with old and new features for training
    """
    model = XGBRegressor(importance_type='total_cover')
    model.fit(df_data, df_target)

    print('>>>>>>>>>>>>>> Start building XGBoost model')
    test_predict = model.predict(df_data_test)
    m_a_e_before = mean_absolute_error(df_target_test.values, test_predict)
    print('Mean absolute error before extra params: {}'.format(m_a_e_before))

    list_column_names = list(df_data.columns)

    dict_imp_features = dict(zip(list_column_names, model.feature_importances_))

    # choose best params and create more stong features
    list_sorted_importance = sorted(model.feature_importances_, reverse=True)
    first_param = [col_name for col_name, imp in dict_imp_features.items() if imp == list_sorted_importance[0]]
    second_param = [col_name for col_name, imp in dict_imp_features.items() if imp == list_sorted_importance[1]]

    # add and multiple values form best params
    str_add_column_name = '({})'.format('__+__'.join(first_param + second_param))
    str_mul_column_name = '({})'.format('__*__'.join(first_param + second_param))
    str_log_column_name = '(log__{})'.format(first_param[0])

    if not (str_add_column_name in list_column_names):
        # Train
        df_data[str_add_column_name] = df_data[first_param].add(df_data[second_param].values).values
        df_data[str_mul_column_name] = df_data[first_param].mul(df_data[second_param].values).values
        # Test
        df_data_test[str_add_column_name] = df_data_test[first_param].add(df_data_test[second_param].values).values
        df_data_test[str_mul_column_name] = df_data_test[first_param].mul(df_data_test[second_param].values).values
    else:
        print('{} exist in data'.format(str_add_column_name))
        print('{} exist in data'.format(str_mul_column_name))

    if not (str_log_column_name in list_column_names):
        # Train
        df_data[str_log_column_name] = np.log(df_data[first_param])
        # Test
        df_data_test[str_log_column_name] = np.log(df_data_test[first_param])
    else:
        print('{} exist in data'.format(str_log_column_name))

    if calculate_score_with_new_features:
        model = XGBRegressor(importance_type='total_cover')
        model.fit(df_data, df_target)

        test_predict = model.predict(df_data_test)
        m_a_e_after = mean_absolute_error(df_target_test.values, test_predict)
        print('Mean absolute error after extra params:  {}'.format(m_a_e_after))

        scoring_improvment = m_a_e_before - m_a_e_after
        return df_data, df_data_test, scoring_improvment

    return df_data, df_data_test
