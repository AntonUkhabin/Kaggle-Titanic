import pandas as pd
from config import config


def load_data(config: config) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load train and test datasets.'''
    
    train_df = pd.read_csv(config.paths.path_to_csv)
    test_df = pd.read_csv(config.paths.path_to_kaggle_test)

    return train_df, test_df