import pandas as pd

from config import config

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer


def build_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for simple Titanic features.'''

    continuous_cols = ['Age']
    discrete_cols_binary = ['IsAlone', 'CabinKnown']
    discrete_cols_count = ['FamilySize', ]
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title', 'Embarked', 'Deck']

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 
            # 'passthrough',
            StandardScaler()
            ),
        ]), continuous_cols),

        ('disc_binary', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 
            # 'passthrough',
            StandardScaler()
            ),
        ]), discrete_cols_binary),

        ('disc_count', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 
            # 'passthrough',
            StandardScaler()
            ),
        ]), discrete_cols_count),

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

    ])

    return preprocessor