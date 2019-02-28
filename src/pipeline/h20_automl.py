import h2o
from h2o.automl import H2OAutoML
from os.path import join
from src.const import columns
from src.const import paths
from src.helper_function import prepare_correct_format_for_result
from src.helper_function import import_files


def run_auto_ml(_df_train,
                _max_runtime_secs=3600,
                _nfolds=10,
                _stopping_metric='mse',
                _sort_metric='mae',
                _exclude_algos=['DeepLearning']):

    print('>>>>>>>>>>>>>> Preparing model and data for model: --{0}min--'.format(_max_runtime_secs / 60))
    hf_train = h2o.H2OFrame(_df_train)

    aml = H2OAutoML(
        max_runtime_secs=_max_runtime_secs,
        nfolds=_nfolds,
        exclude_algos=_exclude_algos,
       # stopping_metric=_stopping_metric,
        sort_metric=_sort_metric
    )

    list_train_columns = list(_df_train.columns)
    list_train_columns.remove(columns.TIME_TO_FAILURE)
    print('>>>>>>>>>>>>>> All set, starting training')
    aml.train(
        x=list_train_columns,
        y=columns.TIME_TO_FAILURE,
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

    print('>>>>>>>>>>>>>> Predict results for test set')
    results_path = join(paths.DIR_PREDICTION, 'results_h2o_{}'.format(_model))
    df_pred = prepare_correct_format_for_result.prepare_result_in_correct_format(df_pred)
    df_pred.to_csv(results_path)


if __name__ == '__main__':
    h2o.init()

    df_train = import_files.import_clean_train_set(path=paths.FILE_TRAIN)

    # run_auto_ml(df_train)

    # test_auto_ml('StackedEnsemble_AllModels_AutoML_20190127_205517', df_test)