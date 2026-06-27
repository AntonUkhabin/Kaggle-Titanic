from sklearn.base import BaseEstimator, TransformerMixin


class TitleExtractor(BaseEstimator, TransformerMixin):
    '''Extract passenger title from Name column.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        features['Title'] = features['Name'].str.extract(r'([A-Za-z]+)\.')

        title_map = {
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs',
            'Capt': 'Other',
            'Col': 'Other',
            'Major': 'Other',
            'Countess': 'Other',
            'Jonkheer': 'Other',
            'Rev': 'Other',
            'Sir': 'Other',
            'Don': 'Other',
            'Lady': 'Other',
            'Dr': 'Other',
        }

        features['Title'] = features['Title'].replace(title_map)

        return features


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    '''Create FamilySize feature from SibSp and Parch.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        features['FamilySize'] = (
            features['SibSp'] + features['Parch'] + 1
        )

        return features


class IsAloneCreator(BaseEstimator, TransformerMixin):
    '''Create IsAlone feature from FamilySize.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        # Create binary feature: True  -> passenger travels alone; False -> passenger travels with family
        # Convert bool values to 0/1 because ML models usually work with numeric features.
        features['IsAlone'] = (
            features['FamilySize'] == 1
        ).astype(int)

        return features