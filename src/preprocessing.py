import pandas as pd

from config import config

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer


def build_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for simple Titanic features.'''

    continuous_cols = ['Age']
    discrete_cols = ['FamilySize', 'IsAlone']
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title']
    custom_cols = []

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 
            # 'passthrough',
            StandardScaler()
            ),
        ]), continuous_cols),

        ('disc', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 
            # 'passthrough',
            StandardScaler()
            ),
        ]), discrete_cols),

        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            # ('scaler', 
            # 'passthrough',
            # # StandardScaler()
            # ),
        ]), ordinal_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_cols),

        ('cust', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('transform', QuantileTransformer(
                output_distribution='normal',
                random_state=config.general.seed,
            )),
        ]), custom_cols),

    ],  
    # verbose=True, # Вывод информации о выполнении шагов
    )

    return preprocessor