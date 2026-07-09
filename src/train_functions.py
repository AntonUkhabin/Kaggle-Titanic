import pandas as pd

from config                     import config
from sklearn.metrics            import accuracy_score
from sklearn.model_selection    import StratifiedKFold, cross_val_score
from sklearn.linear_model       import LogisticRegression
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.pipeline           import Pipeline

from src.preprocessing          import build_preprocessor
from src.feature_engineering    import TitleExtractor, TitleMedianAgeImputer, FamilySizeCreator, IsAloneCreator, CabinKnownCreator, DeckExtractor


def build_model(config):
    '''Build estimator from config.'''

    active_model = config.model.active
    model_params = dict(config.model.models[active_model])

    if active_model == 'logistic_regression':
        return LogisticRegression(**model_params)

    if active_model == 'knn':
        return KNeighborsClassifier(**model_params)

    if config.model.active == 'decision_tree':
        return DecisionTreeClassifier(**model_params)

    raise ValueError(f'Unknown model name: {active_model}')


def build_pipeline(config) -> Pipeline:
    '''Build full machine learning pipeline.'''

    preprocessor = build_preprocessor()
    model = build_model(config)

    pipe = Pipeline([
        ('title_extractor', TitleExtractor()),
        ('title_median_age_imputer', TitleMedianAgeImputer()),
        ('family_size_creator', FamilySizeCreator()),
        ('is_alone_creator', IsAloneCreator()),
        ('cabin_known_creator', CabinKnownCreator()),
        ('deck_extractor', DeckExtractor()),
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    return pipe


def cross_validate_model(train_cv_df, target_col, pipe, config):
    '''Run cross-validation for full sklearn pipeline.'''

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.general.seed,      
    )

    # Split dataframe into features and labels for StratifiedKFold.
    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    scores = cross_val_score(
        estimator=pipe,
        X=features,
        y=labels,
        cv=skf,
        scoring='accuracy',
        verbose=1,
        error_score='raise', # Если в фолде будет ошибка, обучение остановится
        # n_jobs=-10,
    )

    for fold, score in enumerate(scores):
        print(f'Fold: {fold}')
        print(f'Fold Accuracy: {score:.4f}')
        

    return scores.tolist()


def rule_based_model(data: pd.DataFrame) -> list[int]:
    '''Generate predictions using simple rule-based logic.

    Rule:
    - female passengers are predicted as survived;
    - all other passengers are predicted as not survived.
    '''

    preds = []

    for _, row in data.iterrows():
        if row['Sex'] == 'female':
            preds.append(1)
        else:
            preds.append(0)

    return preds


def evaluate_model(labels_true: pd.Series, labels_pred: list[int]) -> float:
    '''Calculate accuracy score for model predictions.'''
    
    score = accuracy_score(labels_true, labels_pred)

    return score


def create_submission(test_df: pd.DataFrame, test_preds: list[int], path_to_submission: str) -> pd.DataFrame:
    '''Create and save Kaggle submission file.'''
    
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_preds
    })

    submission.to_csv(path_to_submission, index=False)

    return submission