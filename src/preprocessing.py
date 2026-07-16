from config                     import config

from sklearn.compose            import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import OneHotEncoder, StandardScaler
from sklearn.base               import BaseEstimator, TransformerMixin


def build_preprocessor(config):
    '''Build preprocessing pipeline for the active model.'''

    if config.model.active == 'catboost':
        categorical_features = list(
            config.model.models.catboost.cat_features
        )

        return CatBoostPreprocessor(
            categorical_features=categorical_features,
        )

    return build_sklearn_preprocessor()


def build_sklearn_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for sklearn models.'''

    continuous_cols = ['Age', 'Fare']
    discrete_cols_binary = ['CabinKnown', 'IsAlone']
    discrete_cols_count = ['FamilySize', ]
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title', 'Deck', 'Embarked']

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            # ('scaler', 
            # # 'passthrough',
            # StandardScaler(),
            # ),
        ]), continuous_cols),

        ('disc_binary', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            # ('scaler', 
            # # 'passthrough',
            # StandardScaler()
            # ),
        ]), discrete_cols_binary),

        ('disc_count', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            # ('scaler', 
            # # 'passthrough',
            # StandardScaler()
            # ),
        ]), discrete_cols_count),

        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), ordinal_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_cols),

    ])

    return preprocessor


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    '''Prepare Titanic features for native CatBoost processing.'''

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features

    def fit(self, features, labels=None):
        self.is_fitted_ = True

        return self

    def transform(self, features):
        features = features.copy()

        numeric_features = [
            'Age',
            'Fare',
            'FamilySize',
            'IsAlone',
            'CabinKnown',
        ]

        selected_features = (
            numeric_features
            + list(self.categorical_features)
        )

        features = features[selected_features]

        features[self.categorical_features] = (
            features[self.categorical_features]
            .fillna('Unknown')
            .astype(str)
        )

        return features