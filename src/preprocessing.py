from config                     import config

from sklearn.compose            import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import OneHotEncoder, StandardScaler
from sklearn.base               import BaseEstimator, TransformerMixin


def build_preprocessor(config):
    '''Build preprocessing pipeline for the active model.'''

    active_model = config.model.active

    if active_model == 'logistic_regression':
        return build_logistic_regression_preprocessor()

    if active_model == 'knn':
        return build_knn_preprocessor()

    if active_model == 'decision_tree':
        return build_tree_preprocessor()

    if active_model == 'random_forest':
        return build_tree_preprocessor()

    if active_model == 'catboost':
        categorical_features = list(
            config.model.models.catboost.cat_features
        )

        return CatBoostPreprocessor(
            categorical_features=categorical_features,
        )

    raise ValueError(f'Unknown preprocessor for model: {active_model}')


def build_logistic_regression_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for Logistic Regression.'''

    continuous_cols = ['Age', 'Fare']
    binary_cols = ['CabinKnown', 'IsAlone']
    count_cols = ['FamilySize']
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title', 'Deck', 'Embarked']

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), continuous_cols),

        ('disc_binary', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), binary_cols),

        ('disc_count', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), count_cols),

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


def build_knn_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for KNN.'''

    continuous_cols = ['Age', 'Fare']
    binary_cols = ['CabinKnown']
    count_cols = ['FamilySize']
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title', 'Deck']

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), continuous_cols),

        ('disc_binary', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), binary_cols),

        ('disc_count', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), count_cols),

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


def build_tree_preprocessor() -> ColumnTransformer:
    '''Build preprocessing pipeline for tree-based sklearn models.'''

    continuous_cols = ['Age', 'Fare']
    binary_cols = ['CabinKnown', 'IsAlone']
    count_cols = ['FamilySize']
    ordinal_cols = ['Pclass']
    categorical_cols = ['Sex', 'Title', 'Deck', 'Embarked']

    preprocessor = ColumnTransformer([
        ('cont', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), continuous_cols),

        ('disc_binary', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), binary_cols),

        ('disc_count', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), count_cols),

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