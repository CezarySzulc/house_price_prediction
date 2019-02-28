from os.path import join


DIR_DATA = join('data')

DIR_DATA_EXTERNAL = join(DIR_DATA, 'external')
DIR_DATA_INTERNAL = join(DIR_DATA, 'internal')

FILE_TRAIN = join(DIR_DATA_EXTERNAL, 'house.csv')

DIR_MODELS = join(DIR_DATA_INTERNAL, 'models')
DIR_PREDICTION = join(DIR_DATA_INTERNAL, 'prediction')
DIR_CLEAN_DATA = join(DIR_DATA_INTERNAL, 'clean_data')

FILE_TRAIN_CLEAR = join(DIR_CLEAN_DATA, 'train_clear.csv')
