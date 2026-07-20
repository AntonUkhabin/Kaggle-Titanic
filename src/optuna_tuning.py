from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.train_functions import build_pipeline


def suggest_model_params(trial, config) -> dict:
    '''Suggest hyperparameters for the active model.'''

    active_model = config.model.active

    if active_model == 'decision_tree':
        return suggest_decision_tree_params(
            trial=trial,
            config=config,
        )

    if active_model == 'random_forest':
        return suggest_random_forest_params(
            trial=trial,
            config=config,
        )

    if active_model == 'catboost':
        return suggest_catboost_params(
            trial=trial,
            config=config,
        )

    raise ValueError(f'Optuna tuning is not configured for: {active_model}')


def suggest_decision_tree_params(trial, config) -> dict:
    '''Suggest Decision Tree hyperparameters for an Optuna trial.'''

    search_space = config.optuna.search_spaces.decision_tree

    params = {
        'model__criterion': trial.suggest_categorical(
            'criterion',
            list(search_space.criterion),
        ),
        'model__max_depth': trial.suggest_int(
            'max_depth',
            search_space.max_depth.low,
            search_space.max_depth.high,
        ),
        'model__min_samples_split': trial.suggest_int(
            'min_samples_split',
            search_space.min_samples_split.low,
            search_space.min_samples_split.high,
        ),
        'model__min_samples_leaf': trial.suggest_int(
            'min_samples_leaf',
            search_space.min_samples_leaf.low,
            search_space.min_samples_leaf.high,
        ),
        'model__max_leaf_nodes': trial.suggest_categorical(
            'max_leaf_nodes',
            list(search_space.max_leaf_nodes),
        ),
        'model__ccp_alpha': trial.suggest_categorical(
            'ccp_alpha',
            list(search_space.ccp_alpha),
        ),
    }

    return params


def suggest_random_forest_params(trial, config) -> dict:
    '''Suggest Random Forest hyperparameters for an Optuna trial.'''

    search_space = config.optuna.search_spaces.random_forest

    params = {
        'model__criterion': trial.suggest_categorical(
            'criterion',
            list(search_space.criterion),
        ),

        'model__n_estimators': trial.suggest_int(
            'n_estimators',
            search_space.n_estimators.low,
            search_space.n_estimators.high,
            step=search_space.n_estimators.step,
        ),

        'model__max_depth': trial.suggest_int(
            'max_depth',
            search_space.max_depth.low,
            search_space.max_depth.high,
        ),

        'model__min_samples_split': trial.suggest_int(
            'min_samples_split',
            search_space.min_samples_split.low,
            search_space.min_samples_split.high,
        ),

        'model__min_samples_leaf': trial.suggest_int(
            'min_samples_leaf',
            search_space.min_samples_leaf.low,
            search_space.min_samples_leaf.high,
        ),

        'model__max_features': trial.suggest_categorical(
            'max_features',
            list(search_space.max_features),
        ),
    }

    return params


def suggest_catboost_params(trial, config) -> dict:
    '''Suggest CatBoost hyperparameters for an Optuna trial.'''

    search_space = config.optuna.search_spaces.catboost

    return {
        'model__depth': trial.suggest_int(
            'depth',
            search_space.depth.low,
            search_space.depth.high,
        ),
        'model__learning_rate': trial.suggest_float(
            'learning_rate',
            search_space.learning_rate.low,
            search_space.learning_rate.high,
            log=True,
        ),
        'model__l2_leaf_reg': trial.suggest_float(
            'l2_leaf_reg',
            search_space.l2_leaf_reg.low,
            search_space.l2_leaf_reg.high,
            log=True,
        ),
        'model__random_strength': trial.suggest_float(
            'random_strength',
            search_space.random_strength.low,
            search_space.random_strength.high,
        ),
    }


def create_catboost_objective(train_cv_df, target_col, config):
    '''Create an Optuna objective with CatBoost early stopping.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.general.seed,
    )

    def objective(trial):
        '''Evaluate one CatBoost hyperparameter combination.'''

        model_params = suggest_catboost_params(trial, config)

        scores = []
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            features_train = features.iloc[train_idx]
            features_val = features.iloc[val_idx]
            labels_train = labels.iloc[train_idx]
            labels_val = labels.iloc[val_idx]

            # Build a fresh pipeline for every fold.
            fold_pipe = build_pipeline(config)
            fold_pipe.set_params(**model_params)

            preprocessing_pipe = fold_pipe[:-1]
            model = fold_pipe.named_steps['model']

            features_train_transformed = preprocessing_pipe.fit_transform(
                features_train,
                labels_train,
            )

            features_val_transformed = preprocessing_pipe.transform(features_val)

            model.fit(
                features_train_transformed,
                labels_train,
                eval_set=(features_val_transformed, labels_val),
                early_stopping_rounds=config.training.early_stopping_rounds,
                use_best_model=True,
            )

            labels_pred = model.predict(features_val_transformed)
            score = accuracy_score(labels_val, labels_pred)
            best_iteration = int(model.tree_count_)

            scores.append(float(score))
            best_iterations.append(best_iteration)

            trial.set_user_attr(f'fold_{fold}_score', float(score))
            trial.set_user_attr(f'fold_{fold}_best_iteration', best_iteration)

        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        mean_best_iteration = float(np.mean(best_iterations))

        trial.set_user_attr('cv_std', cv_std)
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('best_iterations', best_iterations)
        trial.set_user_attr('mean_best_iteration', mean_best_iteration)

        return cv_mean

    return objective


def create_objective(train_cv_df, target_col, base_pipe, config):
    '''Create an Optuna objective for model tuning.'''

    if config.model.active == 'catboost':
        return create_catboost_objective(
            train_cv_df=train_cv_df,
            target_col=target_col,
            config=config,
        )

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.dataloader_params.shuffle,
        random_state=config.general.seed,
    )

    def objective(trial) -> float:
        '''Evaluate one hyperparameter combination.'''

        model_params = suggest_model_params(
            trial=trial,
            config=config,
        )

        # Create a fresh unfitted copy of the full pipeline for this trial.
        trial_pipe = clone(base_pipe)

        # Replace only Decision Tree hyperparameters.
        trial_pipe.set_params(**model_params)

        # Evaluate the full feature engineering, preprocessing and model pipeline inside every cross-validation fold.
        scores = cross_val_score(
            estimator=trial_pipe,
            X=features,
            y=labels,
            cv=skf,
            scoring='accuracy',
            n_jobs=1,
            error_score='raise',
        )

        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        fold_scores = [float(score) for score in scores]

        # Store additional metrics without making them optimization targets.
        trial.set_user_attr('cv_std', cv_std)
        trial.set_user_attr('fold_scores', fold_scores)

        for fold_number, fold_score in enumerate(fold_scores):
            trial.set_user_attr(
                f'fold_{fold_number}_score',
                fold_score,
            )

        return cv_mean

    return objective


def run_optuna_study(
    train_cv_df,
    target_col,
    base_pipe,
    config,
):
    '''Create and run an Optuna study.'''

    sampler = optuna.samplers.TPESampler(
        seed=config.general.seed,
    )

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        direction=config.optuna.direction,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )

    if config.model.active == 'catboost':
        study.enqueue_trial({
            'depth': config.model.models.catboost.depth,
            'learning_rate': config.model.models.catboost.learning_rate,
            'l2_leaf_reg': config.model.models.catboost.l2_leaf_reg,
            'random_strength': config.model.models.catboost.random_strength,
        })

    objective = create_objective(
        train_cv_df=train_cv_df,
        target_col=target_col,
        base_pipe=base_pipe,
        config=config,
    )

    study.optimize(
        objective,
        n_trials=config.optuna.n_trials,
        n_jobs=1,
        show_progress_bar=config.optuna.show_progress_bar,
        gc_after_trial=True,
    )

    return study


def build_trials_dataframe(study) -> pd.DataFrame:
    '''Build a readable dataframe with completed Optuna trials.'''

    trials_df = study.trials_dataframe(
        attrs=(
            'number',
            'value',
            'params',
            'user_attrs',
            'state',
        ),
    )

    trials_df = trials_df.rename(columns={
        'number': 'trial_number',
        'value': 'cv_score',
        'user_attrs_cv_std': 'cv_std',
        'user_attrs_mean_best_iteration': 'mean_best_iteration',
    })

    fold_columns = [
        column
        for column in trials_df.columns
        if column.startswith('user_attrs_fold_')
        and column.endswith('_score')
    ]

    rename_columns = {}

    for column in fold_columns:
        rename_columns[column] = column.removeprefix('user_attrs_')

    trials_df = trials_df.rename(columns=rename_columns)

    # The list form is redundant because every fold is already stored in a separate numeric column.
    if 'user_attrs_fold_scores' in trials_df.columns:
        trials_df = trials_df.drop(
            columns=['user_attrs_fold_scores'],
        )

    parameter_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('params_')
    ])

    fold_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('fold_')
        and column.endswith('_score')
    ])

    ordered_columns = [
        'trial_number',
        'cv_score',
        'cv_std',
        'mean_best_iteration',
        *fold_columns,
        *parameter_columns,
        'state',
    ]

    existing_columns = [
        column
        for column in ordered_columns
        if column in trials_df.columns
    ]

    trials_df = (
        trials_df[existing_columns]
        .sort_values(
            by=['cv_score', 'cv_std'],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return trials_df


def save_optuna_trials(study, output_path) -> pd.DataFrame:
    '''Save all Optuna trials to CSV.'''

    trials_df = build_trials_dataframe(study)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_df.to_csv(
        output_path,
        index=False,
    )

    return trials_df


def save_best_params(study, output_path) -> None:
    '''Save the best Optuna trial parameters to YAML.'''

    best_trial = study.best_trial

    best_params_data = {
        'study_name': study.study_name,
        'trial_number': best_trial.number,
        'cv_score': float(best_trial.value),
        'cv_std': float(best_trial.user_attrs['cv_std']),
        'fold_scores': list(
            best_trial.user_attrs['fold_scores']
        ),
        'params': dict(best_trial.params),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_params_config = OmegaConf.create(
        best_params_data
    )

    OmegaConf.save(
        config=best_params_config,
        f=output_path,
    )


def print_optuna_results(
    study,
    trials_df,
    top_n=10,
) -> None:
    '''Print Optuna study summary and top trials.'''

    best_trial = study.best_trial

    print('\nBest trial:')
    print(f'Trial number: {best_trial.number}')
    print(f'CV Accuracy: {best_trial.value:.4f}')
    print(
        'CV STD: '
        f'{best_trial.user_attrs["cv_std"]:.4f}'
    )

    print('Fold scores:')

    for fold_number, fold_score in enumerate(
        best_trial.user_attrs['fold_scores']
    ):
        print(
            f'Fold {fold_number}: '
            f'{fold_score:.4f}'
        )

    print('\nBest parameters:')

    for param_name, param_value in best_trial.params.items():
        print(f'{param_name}: {param_value}')

    print(f'\nTop {min(top_n, len(trials_df))} trials:')

    parameter_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('params_')
    ])

    columns_to_print = [
        'trial_number',
        'cv_score',
        'cv_std',
        *parameter_columns,
    ]

    existing_columns = [
        column
        for column in columns_to_print
        if column in trials_df.columns
    ]

    print(
        trials_df
        .head(top_n)[existing_columns]
        .to_string(index=False)
    )