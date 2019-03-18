import h2o
from h2o.automl import H2OAutoML
from os.path import join
from sklearn.metrics import mean_absolute_error

from src.const import features
from src.const import paths
from src.helper_function import import_files


def run_auto_ml(_df_train,
                _max_runtime_secs=3600,
                _nfolds=10,
                _stopping_metric='mse',
                _sort_metric='mae',
                _exclude_algos=['DeepLearning']):
    """

    :param _df_train: train DataFrames with all features
    :param _max_runtime_secs: Int, number of seconds that our auto ml will be learn
    :param _nfolds: Int, number of fold
    :param _stopping_metric: Stop metrics for algo
    :param _sort_metric: Sort metrics for algo
    :param _exclude_algos: Excluded algo, in default DeepLearning
    """

    print('>>>>>>>>>>>>>> Preparing model and data for model: --{0}min--'.format(_max_runtime_secs / 60))
    hf_train = h2o.H2OFrame(_df_train)

    aml = H2OAutoML(
        max_runtime_secs=_max_runtime_secs,
        nfolds=_nfolds,
        exclude_algos=_exclude_algos,
        sort_metric=_sort_metric
    )

    list_train_columns = list(_df_train.columns)
    list_train_columns.remove(features.PRICE)

    print('>>>>>>>>>>>>>> All set, starting training')
    aml.train(
        x=list_train_columns,
        y=features.PRICE,
        training_frame=hf_train
    )

    print('>>>>>>>>>>>>>> Finished training')
    saved_path = h2o.save_model(model=aml.leader, path=paths.DIR_MODELS, force=True)

    print('>>>>>>>>>>>>>> Model saved on path: {}'.format(saved_path))

    print('>>>>>>>>>>>>>> Models saved')
    print('\n\n')
    print(aml.leaderboard.head())


def test_auto_ml(_model, _df_test):
    print('>>>>>>>>>>>>>> Import model and test set')

    model_path = join(paths.DIR_MODELS, _model)
    model = h2o.load_model(model_path)
    hf_test = h2o.H2OFrame(_df_test)

    print('>>>>>>>>>>>>>> Predict results for test set')
    df_pred = model.predict(hf_test).as_data_frame()

    print('>>>>>>>>>>>>>> Calculate mean absolute error')
    m_a_e = mean_absolute_error(df_pred.values, _df_test[features.PRICE].values)
    print('Mean absolute error:  {}'.format(m_a_e))


if __name__ == '__main__':
    h2o.init()

    df_train = import_files.import_auto_train_data()
    df_test = import_files.import_auto_test_data()

    # run_auto_ml(df_train)

    test_auto_ml('XGBoost_3_AutoML_20190317_222432', df_test)
