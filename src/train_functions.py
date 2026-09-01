import pandas as pd
import numpy as np

from sklearn.metrics            import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection    import StratifiedKFold
from sklearn.linear_model       import LogisticRegression
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.pipeline           import Pipeline
from sklearn.ensemble           import RandomForestClassifier
from catboost                   import CatBoostClassifier
from lightgbm                   import LGBMClassifier, early_stopping, log_evaluation
from xgboost                    import XGBClassifier

from src.preprocessing          import build_preprocessor
from src.feature_engineering    import TitleExtractor, TitleMedianAgeImputer, FamilySizeCreator, IsAloneCreator, CabinKnownCreator, DeckExtractor


def build_model(config, categorical_features=None):
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
        if categorical_features is None:
            raise ValueError('Categorical features are required for CatBoost.')

        return CatBoostClassifier(cat_features=list(categorical_features), **model_params)

    if active_model == 'lightgbm':
        return LGBMClassifier(**model_params)

    if active_model == 'xgboost':
        return XGBClassifier(**model_params)

    raise ValueError(f'Unknown model name: {active_model}')



def build_feature_pipeline(config) -> Pipeline:
    '''Build feature engineering and preprocessing pipeline without a model.'''

    preprocessor = build_preprocessor(config)

    feature_pipeline = Pipeline([
        ('title_extractor', TitleExtractor()),
        ('title_median_age_imputer', TitleMedianAgeImputer()),
        ('family_size_creator', FamilySizeCreator()),
        ('is_alone_creator', IsAloneCreator()),
        ('cabin_known_creator', CabinKnownCreator()),
        ('deck_extractor', DeckExtractor()),
        ('preprocessor', preprocessor),
    ])

    return feature_pipeline


def build_pipeline(config) -> Pipeline:
    '''Build full machine learning pipeline.'''

    feature_pipeline = build_feature_pipeline(config)
    preprocessor = feature_pipeline.named_steps['preprocessor']
    categorical_features = getattr(preprocessor, 'categorical_features', None)
    model = build_model(config, categorical_features=categorical_features)

    pipe = Pipeline(feature_pipeline.steps + [('model', model)])

    return pipe


def cross_validate_standard(train_cv_df, target_col, config, keep_fold_models=False):
    '''Run standard cross-validation, generate OOF predictions and fit the final pipeline.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.training.fold_seed,
    )

    scores = []
    fold_models = []
    oof_probabilities = np.zeros(len(features))

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]
        labels_train = labels.iloc[train_idx]
        labels_val = labels.iloc[val_idx]

        fold_pipe = build_pipeline(config)
        fold_pipe.fit(features_train, labels_train)

        validation_probabilities = fold_pipe.predict_proba(features_val)[:, 1]
        labels_pred = fold_pipe.predict(features_val)

        # Restore original row order; each prediction comes from a model
        # that was trained without this validation fold.
        oof_probabilities[val_idx] = validation_probabilities
        score = accuracy_score(labels_val, labels_pred)

        scores.append(float(score))

        if keep_fold_models:
            fold_models.append(fold_pipe)

        print(f'Fold: {fold}')
        print(f'Fold Accuracy: {score:.4f}')

    if keep_fold_models:
        result_pipe = fold_models[0]
    else:
        result_pipe = build_pipeline(config)
        result_pipe.fit(features, labels)

    return scores, result_pipe, fold_models, oof_probabilities


def predict_with_pipeline(features, pipe):
    '''Generate predictions with a fitted pipeline.'''

    probabilities = pipe.predict_proba(features)[:, 1]
    predictions = pipe.predict(features).astype(int)

    return predictions, probabilities


def predict_with_pipeline_ensemble(features, fold_models, threshold=0.5):
    '''Generate predictions by averaging fitted pipeline probabilities.'''

    fold_probabilities = [
        fold_pipe.predict_proba(features)[:, 1]
        for fold_pipe in fold_models
    ]

    mean_probabilities = np.mean(fold_probabilities, axis=0)
    predictions = (mean_probabilities > threshold).astype(int)

    return predictions, mean_probabilities


def fit_model_with_early_stopping(model, features_train, labels_train, features_val, labels_val, config):
    '''Fit model with validation-based early stopping.'''

    active_model = config.model.active

    if active_model == 'catboost':
        model.fit(
            features_train,
            labels_train,
            eval_set=(features_val, labels_val),
            early_stopping_rounds=config.training.early_stopping_rounds,
            use_best_model=True,
        )

        return model

    if active_model == 'lightgbm':
        callbacks = [
            early_stopping(
                stopping_rounds=config.training.early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
            log_evaluation(period=0),
        ]

        model.fit(
            features_train,
            labels_train,
            eval_X=features_val,
            eval_y=labels_val,
            eval_names=['validation'],
            callbacks=callbacks,
        )

        return model

    if active_model == 'xgboost':
        model.set_params(
            early_stopping_rounds=config.training.early_stopping_rounds,
        )

        model.fit(
            features_train,
            labels_train,
            eval_set=[(features_val, labels_val)],
            verbose=False,
        )

        return model

    raise ValueError(f'Early stopping is not supported for model: {active_model}')


def report_early_stopping_results(model, score, fold, config):
    '''Print early stopping results and return retained tree count.'''

    active_model = config.model.active

    if active_model == 'catboost':
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
        print(f'Fold Accuracy: {score:.4f}')
        print(f'Best Accuracy: {best_accuracy:.4f}')
        print(f'Best Accuracy iteration: {best_accuracy_iteration + 1}')
        print(f'Best Logloss iteration: {best_logloss_iteration + 1}')
        print(f'Best validation Logloss: {best_logloss:.4f}')
        print(f'Iterations actually run: {iterations_run}')
        print(f'Trees retained: {trees_built}')

        return trees_built

    if active_model == 'lightgbm':
        metric_name = config.model.models.lightgbm.metric
        validation_metric = model.evals_result_['validation'][metric_name]
        best_iteration = model.best_iteration_
        best_metric = model.best_score_['validation'][metric_name]
        iterations_run = len(validation_metric)
        trees_built = model.n_estimators_

        print(f'Fold: {fold}')
        print(f'Fold Accuracy: {score:.4f}')
        print(f'Early stopping metric: {metric_name}')
        print(f'Best validation {metric_name}: {best_metric:.4f}')
        print(f'Best {metric_name} iteration: {best_iteration}')
        print(f'Iterations actually run: {iterations_run}')
        print(f'Trees retained: {trees_built}')

        return trees_built

    if active_model == 'xgboost':
        metric_name = config.model.models.xgboost.eval_metric
        validation_metric = model.evals_result()['validation_0'][metric_name]
        best_iteration = model.best_iteration
        best_metric = float(model.best_score)
        iterations_run = len(validation_metric)
        trees_used = best_iteration + 1

        print(f'Fold: {fold}')
        print(f'Fold Accuracy: {score:.4f}')
        print(f'Early stopping metric: {metric_name}')
        print(f'Best validation {metric_name}: {best_metric:.4f}')
        print(f'Best {metric_name} iteration: {trees_used}')
        print(f'Iterations actually run: {iterations_run}')
        print(f'Trees used for prediction: {trees_used}')

        return trees_used

    raise ValueError(f'Early stopping results are not supported for model: {active_model}')


def cross_validate_model_with_early_stopping(train_cv_df, target_col, config):
    '''Run cross-validation with fold-specific early stopping.'''

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.training.fold_seed,
    )

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    scores = []
    best_iterations = []
    fold_models = []
    oof_probabilities = np.zeros(len(features))

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
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
        features_train_transformed = preprocessing_pipe.fit_transform(features_train, labels_train)

        # Apply the fitted transformations to fold_validation.
        features_val_transformed = preprocessing_pipe.transform(features_val)

        model = fit_model_with_early_stopping(
            model=model,
            features_train=features_train_transformed,
            labels_train=labels_train,
            features_val=features_val_transformed,
            labels_val=labels_val,
            config=config,
        )

        validation_probabilities = model.predict_proba(features_val_transformed)[:, 1]
        oof_probabilities[val_idx] = validation_probabilities
        labels_pred = model.predict(features_val_transformed)

        score = accuracy_score(labels_val, labels_pred)

        trees_built = report_early_stopping_results(
            model=model,
            score=score,
            fold=fold,
            config=config,
        )

        scores.append(score)
        best_iterations.append(trees_built)
        fold_models.append((preprocessing_pipe, model))

    return scores, best_iterations, fold_models, oof_probabilities


def predict_with_early_stopping_ensemble(features, fold_models, threshold=0.5):
    '''Generate predictions by averaging early-stopping fold-model probabilities.'''

    fold_probabilities = []

    for preprocessing_pipe, model in fold_models:
        features_transformed = preprocessing_pipe.transform(features)
        probabilities = model.predict_proba(features_transformed)[:, 1]
        fold_probabilities.append(probabilities)

    mean_probabilities = np.mean(fold_probabilities, axis=0)
    predictions = (mean_probabilities > threshold).astype(int)

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


def calculate_classification_metrics(labels, predictions, probabilities):
    '''Calculate classification and probability metrics.'''

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(labels, probabilities),
        'pr_auc': average_precision_score(labels, probabilities),
        'logloss': log_loss(labels, probabilities),
        'brier': brier_score_loss(labels, probabilities),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def create_submission(test_df: pd.DataFrame, test_preds: list[int], path_to_submission: str) -> pd.DataFrame:
    '''Create and save Kaggle submission file.'''
    
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_preds
    })

    submission.to_csv(path_to_submission, index=False)

    return submission