from sklearn.base import BaseEstimator, TransformerMixin


class TitleExtractor(BaseEstimator, TransformerMixin):
    '''Extract passenger title from Name column.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        # Extract title from passenger name, for example: 'Mr.', 'Mrs.', 'Miss.'.
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

        # Group rare titles to reduce noise and category sparsity.
        features['Title'] = features['Title'].replace(title_map)

        return features


class TitleMedianAgeImputer(BaseEstimator, TransformerMixin):
    '''Fill missing Age values using median Age by mapped Title.'''

    def fit(self, features, labels=None):
        # Learn Age medians from training data only to avoid data leakage.
        self.age_medians_ = features.groupby('Title')['Age'].median()
        self.global_median_ = features['Age'].median()

        return self

    def transform(self, features):
        features = features.copy()

        missing_age_mask = features['Age'].isna()

        # Fill missing Age with Title-specific median.
        # Fallback to global median if an unseen Title appears.
        features.loc[missing_age_mask, 'Age'] = (
            features.loc[missing_age_mask, 'Title']
            .map(self.age_medians_)
            .fillna(self.global_median_)
        )

        return features


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    '''Create FamilySize feature from SibSp and Parch.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        features['FamilySize'] = (features['SibSp'] + features['Parch'] + 1)

        return features


class IsAloneCreator(BaseEstimator, TransformerMixin):
    '''Create IsAlone feature from FamilySize.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        # Create binary feature: True  -> passenger travels alone; False -> passenger travels with family
        # Convert bool values to 0/1 because ML models usually work with numeric features.
        features['IsAlone'] = (features['FamilySize'] == 1).astype(int)

        return features


class CabinKnownCreator(BaseEstimator, TransformerMixin):
    '''Create CabinKnown from Cabin column.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        # Create binary feature: 1 -> Cabin value is known; 0 -> Cabin value is missing
        features['CabinKnown'] = (features['Cabin'].notna()).astype(int)

        return features


class DeckExtractor(BaseEstimator, TransformerMixin):
    '''Extract Deck feature from Cabin column.'''

    def fit(self, features, labels=None):
        return self

    def transform(self, features):
        features = features.copy()

        # Extract the first letter from Cabin.
        # Example: 'C85' -> 'C', missing Cabin -> 'Unknown'.
        features['Deck'] = (features['Cabin'].str[0].fillna('Unknown'))

        return features