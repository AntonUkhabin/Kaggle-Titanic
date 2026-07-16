import pandas as pd
import numpy as np

from config                     import config
from sklearn.metrics            import accuracy_score
from sklearn.model_selection    import StratifiedKFold, cross_val_score
from sklearn.linear_model       import LogisticRegression
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.pipeline           import Pipeline
from sklearn.ensemble           import RandomForestClassifier
from catboost                   import CatBoostClassifier

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

    if active_model == 'decision_tree':
        return DecisionTreeClassifier(**model_params)

    if active_model == 'random_forest':
        return RandomForestClassifier(**model_params)

    if active_model == 'catboost':
        return CatBoostClassifier(**model_params)

    raise ValueError(f'Unknown model name: {active_model}')


def build_pipeline(config) -> Pipeline:
    '''Build full machine learning pipeline.'''

    preprocessor = build_preprocessor(config)
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


def cross_validate_model_with_early_stopping(train_cv_df, target_col, config):
    '''Run cross-validation with fold-specific early stopping.'''

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.general.seed,
    )

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    scores = []
    best_iterations = []
    fold_models = []
    oof_probabilities = np.zeros(len(features))

    best_accuracy_scores = []
    best_accuracy_iterations = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(features, labels)
    ):
        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]

        labels_train = labels.iloc[train_idx]
        labels_val = labels.iloc[val_idx]

        # Build a new unfitted pipeline for every fold.
        fold_pipe = build_pipeline(config)

        # Separate preprocessing from the final estimator.
        preprocessing_pipe = fold_pipe[:-1]
        model = fold_pipe.named_steps['model']

        # Fit feature engineering and preprocessing only on fold_train.
        features_train_transformed = (
            preprocessing_pipe.fit_transform(
                features_train,
                labels_train,
            )
        )

        # Apply the fitted transformations to fold_validation.
        features_val_transformed = (
            preprocessing_pipe.transform(
                features_val,
            )
        )

        model.fit(
            features_train_transformed,
            labels_train,
            eval_set=(
                features_val_transformed,
                labels_val,
            ),
            early_stopping_rounds=(
                config.training.early_stopping_rounds
            ),
            use_best_model=True,
        )

        validation_probabilities = model.predict_proba(features_val_transformed)[:, 1]
        oof_probabilities[val_idx] = validation_probabilities
        labels_pred = model.predict(features_val_transformed)

        score = accuracy_score(
            labels_val,
            labels_pred,
        )

        best_iteration = model.get_best_iteration()
        trees_built = model.tree_count_

        evals_result = model.get_evals_result()

        validation_logloss = evals_result['validation']['Logloss']
        best_logloss_iteration = int(np.argmin(validation_logloss))
        best_logloss = validation_logloss[best_logloss_iteration]
        iterations_run = len(evals_result['validation']['Accuracy'])

        validation_accuracy = evals_result['validation'].get('Accuracy')

        if validation_accuracy is None:
            available_metrics = list(evals_result['validation'])
            raise KeyError(f'Accuracy is missing. Available metrics: {available_metrics}')

        best_accuracy_iteration = int(np.argmax(validation_accuracy))
        best_accuracy = validation_accuracy[best_accuracy_iteration]

        print(f'Fold: {fold}')
        print(f'Accuracy at best Logloss: {score:.4f}')
        print(f'Best Accuracy: {best_accuracy:.4f}')
        print(f'Best Accuracy iteration: {best_accuracy_iteration + 1}')
        print(f'Best Logloss iteration: {best_iteration + 1}')
        print(f'Best validation Logloss: {best_logloss:.4f}')
        print(f'Iterations actually run: {iterations_run}')
        print(f'Trees retained: {trees_built}')

        scores.append(score)
        best_iterations.append(trees_built)
        fold_models.append((preprocessing_pipe, model))

        best_accuracy_scores.append(best_accuracy)
        best_accuracy_iterations.append(best_accuracy_iteration + 1)

    return scores, best_iterations, fold_models, oof_probabilities


def predict_with_catboost_ensemble(features, fold_models, threshold=0.5):
    '''Generate predictions by averaging fold-model probabilities.'''

    fold_probabilities = []

    for preprocessing_pipe, model in fold_models:
        features_transformed = preprocessing_pipe.transform(features)
        probabilities = model.predict_proba(features_transformed)[:, 1]
        fold_probabilities.append(probabilities)

    mean_probabilities = np.mean(fold_probabilities, axis=0)
    predictions = (mean_probabilities >= threshold).astype(int)

    return predictions, mean_probabilities


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