import pandas as pd

from config import config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.feature_engineering import TitleExtractor, TitleMedianAgeImputer, FamilySizeCreator, IsAloneCreator, CabinKnownCreator, DeckExtractor


def build_logreg_pipeline(config) -> Pipeline:
    '''Build Logistic Regression pipeline.'''

    preprocessor = build_preprocessor()

    pipe = Pipeline([
        ('title_extractor', TitleExtractor()),
        ('title_median_age_imputer', TitleMedianAgeImputer()),
        ('family_size_creator', FamilySizeCreator()),
        ('is_alone_creator', IsAloneCreator()),
        ('cabin_known_creator', CabinKnownCreator()),
        ('deck_extractor', DeckExtractor()),
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            max_iter=500,
            random_state=config.general.seed,
            C=1,
            solver='lbfgs',
            l1_ratio=0, # 0->l2, 1->l1
            class_weight=None,
            # verbose=10, # можно поставить 1 или 10
            ),
        ),
    ], 
    # verbose=True,
    )

    return pipe


def log_logreg_model(pipe) -> None:
    '''Print fitted Logistic Regression information.'''

    model = pipe.named_steps['model']

    print(f'Number of iterations: {model.n_iter_[0]}')

    feature_names = pipe.named_steps[
        'preprocessor'
    ].get_feature_names_out()

    coef = model.coef_[0]

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coef,
        'abs_coef': abs(coef),
    }).sort_values('abs_coef', ascending=False)

    print(coef_df)


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