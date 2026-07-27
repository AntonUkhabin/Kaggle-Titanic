from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.train_functions import build_pipeline, fit_model_with_early_stopping


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

    if active_model == 'lightgbm':
        return suggest_lightgbm_params(
            trial=trial,
            config=config,
        )

    if active_model == 'xgboost':
        return suggest_xgboost_params(
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


def suggest_lightgbm_params(trial, config) -> dict:
    '''Suggest LightGBM hyperparameters for an Optuna trial.'''

    search_space = config.optuna.search_spaces.lightgbm

    max_depth = trial.suggest_int(
        'max_depth',
        search_space.max_depth.low,
        search_space.max_depth.high,
    )

    max_num_leaves = min(
        search_space.num_leaves.high,
        2 ** max_depth,
    )

    subsample_freq = trial.suggest_categorical(
        'subsample_freq',
        list(search_space.subsample_freq),
    )

    params = {
        'model__learning_rate': trial.suggest_float(
            'learning_rate',
            search_space.learning_rate.low,
            search_space.learning_rate.high,
            log=True,
        ),

        'model__max_depth': max_depth,

        'model__num_leaves': trial.suggest_int(
            'num_leaves',
            search_space.num_leaves.low,
            max_num_leaves,
        ),

        'model__min_child_samples': trial.suggest_int(
            'min_child_samples',
            search_space.min_child_samples.low,
            search_space.min_child_samples.high,
            step=search_space.min_child_samples.step,
        ),

        'model__min_child_weight': trial.suggest_float(
            'min_child_weight',
            search_space.min_child_weight.low,
            search_space.min_child_weight.high,
            log=True,
        ),

        'model__min_split_gain': trial.suggest_float(
            'min_split_gain',
            search_space.min_split_gain.low,
            search_space.min_split_gain.high,
        ),

        'model__subsample_freq': subsample_freq,

        'model__colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            search_space.colsample_bytree.low,
            search_space.colsample_bytree.high,
        ),

        'model__reg_alpha': trial.suggest_categorical(
            'reg_alpha',
            list(search_space.reg_alpha),
        ),

        'model__reg_lambda': trial.suggest_categorical(
            'reg_lambda',
            list(search_space.reg_lambda),
        ),

        'model__cat_smooth': trial.suggest_float(
            'cat_smooth',
            search_space.cat_smooth.low,
            search_space.cat_smooth.high,
        ),

        'model__cat_l2': trial.suggest_float(
            'cat_l2',
            search_space.cat_l2.low,
            search_space.cat_l2.high,
            log=True,
        ),

        'model__min_data_per_group': trial.suggest_int(
            'min_data_per_group',
            search_space.min_data_per_group.low,
            search_space.min_data_per_group.high,
            step=search_space.min_data_per_group.step,
        ),

        'model__max_cat_to_onehot': trial.suggest_categorical(
            'max_cat_to_onehot',
            list(search_space.max_cat_to_onehot),
        ),

        'model__max_cat_threshold': trial.suggest_categorical(
            'max_cat_threshold',
            list(search_space.max_cat_threshold),
        ),

        'model__max_bin': trial.suggest_categorical(
            'max_bin',
            list(search_space.max_bin),
        ),
    }

    if subsample_freq == 0:
        params['model__subsample'] = 1.0
    else:
        params['model__subsample'] = trial.suggest_float(
            'subsample',
            search_space.subsample.low,
            search_space.subsample.high,
        )

    return params


def suggest_xgboost_params(trial, config) -> dict:
    '''Suggest XGBoost hyperparameters for an Optuna trial.'''

    search_space = config.optuna.search_spaces.xgboost

    return {
        'model__learning_rate': trial.suggest_float(
            'learning_rate',
            search_space.learning_rate.low,
            search_space.learning_rate.high,
            log=True,
        ),

        'model__max_depth': trial.suggest_int(
            'max_depth',
            search_space.max_depth.low,
            search_space.max_depth.high,
        ),

        'model__subsample': trial.suggest_float(
            'subsample',
            search_space.subsample.low,
            search_space.subsample.high,
        ),

        'model__colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            search_space.colsample_bytree.low,
            search_space.colsample_bytree.high,
        ),

        'model__reg_alpha': trial.suggest_float(
            'reg_alpha',
            search_space.reg_alpha.low,
            search_space.reg_alpha.high,
        ),

        'model__reg_lambda': trial.suggest_float(
            'reg_lambda',
            search_space.reg_lambda.low,
            search_space.reg_lambda.high,
            log=True,
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


def create_lightgbm_objective(train_cv_df, target_col, config):
    '''Create a multi-seed Optuna objective with LightGBM early stopping.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]
    fold_seeds = list(config.optuna.fold_seeds)

    def objective(trial) -> float:
        '''Evaluate one LightGBM hyperparameter combination.'''

        model_params = suggest_lightgbm_params(
            trial=trial,
            config=config,
        )

        all_scores = []
        all_best_iterations = []
        seed_cv_means = []
        seed_cv_stds = []

        for seed_index, fold_seed in enumerate(fold_seeds):
            skf = StratifiedKFold(
                n_splits=config.split.n_splits,
                shuffle=config.dataloader_params.shuffle,
                random_state=fold_seed,
            )

            seed_scores = []
            seed_best_iterations = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                features_train = features.iloc[train_idx]
                features_val = features.iloc[val_idx]
                labels_train = labels.iloc[train_idx]
                labels_val = labels.iloc[val_idx]

                fold_pipe = build_pipeline(config)
                fold_pipe.set_params(**model_params)

                preprocessing_pipe = fold_pipe[:-1]
                model = fold_pipe.named_steps['model']

                features_train_transformed = preprocessing_pipe.fit_transform(
                    features_train,
                    labels_train,
                )
                features_val_transformed = preprocessing_pipe.transform(features_val)

                model = fit_model_with_early_stopping(
                    model=model,
                    features_train=features_train_transformed,
                    labels_train=labels_train,
                    features_val=features_val_transformed,
                    labels_val=labels_val,
                    config=config,
                )

                labels_pred = model.predict(features_val_transformed)
                score = float(accuracy_score(labels_val, labels_pred))
                best_iteration = int(model.best_iteration_)

                seed_scores.append(score)
                seed_best_iterations.append(best_iteration)
                all_scores.append(score)
                all_best_iterations.append(best_iteration)

                global_fold = seed_index * config.split.n_splits + fold
                trial.set_user_attr(f'fold_{global_fold}_score', score)
                trial.set_user_attr(f'fold_{global_fold}_best_iteration', best_iteration)

            seed_cv_mean = float(np.mean(seed_scores))
            seed_cv_std = float(np.std(seed_scores))

            seed_cv_means.append(seed_cv_mean)
            seed_cv_stds.append(seed_cv_std)

            trial.set_user_attr(f'seed_{seed_index}_fold_seed', int(fold_seed))
            trial.set_user_attr(f'seed_{seed_index}_cv_mean', seed_cv_mean)
            trial.set_user_attr(f'seed_{seed_index}_cv_std', seed_cv_std)
            trial.set_user_attr(f'seed_{seed_index}_best_iterations', seed_best_iterations)

        cv_mean = float(np.mean(seed_cv_means))
        cv_std = float(np.std(all_scores))
        seed_cv_std = float(np.std(seed_cv_means))
        mean_best_iteration = float(np.mean(all_best_iterations))
        median_best_iteration = float(np.median(all_best_iterations))

        trial.set_user_attr('cv_std', cv_std)
        trial.set_user_attr('seed_cv_std', seed_cv_std)
        trial.set_user_attr('fold_scores', all_scores)
        trial.set_user_attr('seed_cv_means', seed_cv_means)
        trial.set_user_attr('seed_cv_stds', seed_cv_stds)
        trial.set_user_attr('best_iterations', all_best_iterations)
        trial.set_user_attr('mean_best_iteration', mean_best_iteration)
        trial.set_user_attr('median_best_iteration', median_best_iteration)

        return cv_mean

    return objective


def create_xgboost_objective(train_cv_df, target_col, config):
    '''Create a multi-seed Optuna objective with XGBoost early stopping.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]
    fold_seeds = list(config.optuna.fold_seeds)

    def objective(trial) -> float:
        '''Evaluate one XGBoost hyperparameter combination.'''

        model_params = suggest_xgboost_params(
            trial=trial,
            config=config,
        )

        all_scores = []
        all_best_iterations = []
        seed_cv_means = []
        seed_cv_stds = []

        for seed_index, fold_seed in enumerate(fold_seeds):
            skf = StratifiedKFold(
                n_splits=config.split.n_splits,
                shuffle=config.dataloader_params.shuffle,
                random_state=fold_seed,
            )

            seed_scores = []
            seed_best_iterations = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                features_train = features.iloc[train_idx]
                features_val = features.iloc[val_idx]
                labels_train = labels.iloc[train_idx]
                labels_val = labels.iloc[val_idx]

                fold_pipe = build_pipeline(config)
                fold_pipe.set_params(**model_params)

                preprocessing_pipe = fold_pipe[:-1]
                model = fold_pipe.named_steps['model']

                features_train_transformed = preprocessing_pipe.fit_transform(
                    features_train,
                    labels_train,
                )
                features_val_transformed = preprocessing_pipe.transform(features_val)

                model = fit_model_with_early_stopping(
                    model=model,
                    features_train=features_train_transformed,
                    labels_train=labels_train,
                    features_val=features_val_transformed,
                    labels_val=labels_val,
                    config=config,
                )

                labels_pred = model.predict(features_val_transformed)
                score = float(accuracy_score(labels_val, labels_pred))
                best_iteration = int(model.best_iteration + 1)

                seed_scores.append(score)
                seed_best_iterations.append(best_iteration)
                all_scores.append(score)
                all_best_iterations.append(best_iteration)

                global_fold = seed_index * config.split.n_splits + fold
                trial.set_user_attr(f'fold_{global_fold}_score', score)
                trial.set_user_attr(f'fold_{global_fold}_best_iteration', best_iteration)

            seed_cv_mean = float(np.mean(seed_scores))
            seed_cv_std = float(np.std(seed_scores))

            seed_cv_means.append(seed_cv_mean)
            seed_cv_stds.append(seed_cv_std)

            trial.set_user_attr(f'seed_{seed_index}_fold_seed', int(fold_seed))
            trial.set_user_attr(f'seed_{seed_index}_cv_mean', seed_cv_mean)
            trial.set_user_attr(f'seed_{seed_index}_cv_std', seed_cv_std)
            trial.set_user_attr(f'seed_{seed_index}_best_iterations', seed_best_iterations)

        cv_mean = float(np.mean(seed_cv_means))
        cv_std = float(np.std(all_scores))
        seed_cv_std = float(np.std(seed_cv_means))
        mean_best_iteration = float(np.mean(all_best_iterations))
        median_best_iteration = float(np.median(all_best_iterations))

        trial.set_user_attr('cv_std', cv_std)
        trial.set_user_attr('seed_cv_std', seed_cv_std)
        trial.set_user_attr('fold_scores', all_scores)
        trial.set_user_attr('seed_cv_means', seed_cv_means)
        trial.set_user_attr('seed_cv_stds', seed_cv_stds)
        trial.set_user_attr('best_iterations', all_best_iterations)
        trial.set_user_attr('mean_best_iteration', mean_best_iteration)
        trial.set_user_attr('median_best_iteration', median_best_iteration)

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

    if config.model.active == 'lightgbm':
        return create_lightgbm_objective(
            train_cv_df=train_cv_df,
            target_col=target_col,
            config=config,
        )

    if config.model.active == 'xgboost':
        return create_xgboost_objective(
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

    if config.model.active == 'lightgbm':
        for control_trial in config.optuna.enqueued_trials.lightgbm:
            study.enqueue_trial(
                OmegaConf.to_container(
                    control_trial,
                    resolve=True,
                )
            )

    if config.model.active == 'xgboost':
        for control_trial in config.optuna.enqueued_trials.xgboost:
            study.enqueue_trial(
                OmegaConf.to_container(
                    control_trial,
                    resolve=True,
                )
            )

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
        'user_attrs_seed_cv_std': 'seed_cv_std',
        'user_attrs_mean_best_iteration': 'mean_best_iteration',
        'user_attrs_median_best_iteration': 'median_best_iteration',
    })

    rename_columns = {}

    for column in trials_df.columns:
        if column.startswith('user_attrs_fold_') and column.endswith('_score'):
            rename_columns[column] = column.removeprefix('user_attrs_')

        if column.startswith('user_attrs_seed_') and (
            column.endswith('_fold_seed')
            or column.endswith('_cv_mean')
            or column.endswith('_cv_std')
        ):
            rename_columns[column] = column.removeprefix('user_attrs_')

    trials_df = trials_df.rename(columns=rename_columns)

    redundant_columns = [
        'user_attrs_fold_scores',
        'user_attrs_seed_cv_means',
        'user_attrs_seed_cv_stds',
        'user_attrs_best_iterations',
    ]

    redundant_columns.extend([
        column
        for column in trials_df.columns
        if column.startswith('user_attrs_seed_')
        and column.endswith('_best_iterations')
    ])

    existing_redundant_columns = [
        column
        for column in redundant_columns
        if column in trials_df.columns
    ]

    if existing_redundant_columns:
        trials_df = trials_df.drop(columns=existing_redundant_columns)

    parameter_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('params_')
    ])

    seed_metric_order = {
        'fold_seed': 0,
        'cv_mean': 1,
        'cv_std': 2,
    }

    seed_columns = sorted(
        [
            column
            for column in trials_df.columns
            if column.startswith('seed_')
            and column != 'seed_cv_std'
        ],
        key=lambda column: (
            int(column.split('_')[1]),
            seed_metric_order['_'.join(column.split('_')[2:])],
        ),
    )

    fold_columns = sorted(
        [
            column
            for column in trials_df.columns
            if column.startswith('fold_')
            and column.endswith('_score')
        ],
        key=lambda column: int(column.split('_')[1]),
    )

    ordered_columns = [
        'trial_number',
        'cv_score',
        'seed_cv_std',
        'cv_std',
        *seed_columns,
        'mean_best_iteration',
        'median_best_iteration',
        *fold_columns,
        *parameter_columns,
        'state',
    ]

    existing_columns = [
        column
        for column in ordered_columns
        if column in trials_df.columns
    ]

    sort_columns = ['cv_score']
    ascending = [False]

    for column in ['seed_cv_std', 'cv_std']:
        if column in trials_df.columns:
            sort_columns.append(column)
            ascending.append(True)

    trials_df = (
        trials_df[existing_columns]
        .sort_values(
            by=sort_columns,
            ascending=ascending,
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
    seed_cv_means = best_trial.user_attrs.get('seed_cv_means', [])
    seed_results = []

    for seed_index, seed_cv_mean in enumerate(seed_cv_means):
        seed_results.append({
            'fold_seed': int(best_trial.user_attrs[f'seed_{seed_index}_fold_seed']),
            'cv_score': float(seed_cv_mean),
            'cv_std': float(best_trial.user_attrs[f'seed_{seed_index}_cv_std']),
            'best_iterations': list(best_trial.user_attrs[f'seed_{seed_index}_best_iterations']),
        })

    best_params_data = {
        'study_name': study.study_name,
        'trial_number': best_trial.number,
        'cv_score': float(best_trial.value),
        'seed_cv_std': float(best_trial.user_attrs.get('seed_cv_std', 0.0)),
        'cv_std': float(best_trial.user_attrs['cv_std']),
        'seed_results': seed_results,
        'fold_scores': list(best_trial.user_attrs['fold_scores']),
        'mean_best_iteration': float(best_trial.user_attrs['mean_best_iteration']),
        'median_best_iteration': float(best_trial.user_attrs.get('median_best_iteration', 0.0)),
        'params': dict(best_trial.params),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_params_config = OmegaConf.create(best_params_data)

    OmegaConf.save(
        config=best_params_config,
        f=output_path,
    )


def print_optuna_results(study, trials_df, top_n=10) -> None:
    '''Print Optuna study summary and top trials.'''

    best_trial = study.best_trial

    print('\nBest trial:')
    print(f'Trial number: {best_trial.number}')
    print(f'Multi-seed CV Accuracy: {best_trial.value:.4f}')
    print(f'Seed CV STD: {best_trial.user_attrs.get("seed_cv_std", 0.0):.4f}')
    print(f'All folds CV STD: {best_trial.user_attrs["cv_std"]:.4f}')
    print(f'Mean best iteration: {best_trial.user_attrs.get("mean_best_iteration", 0.0):.1f}')
    print(f'Median best iteration: {best_trial.user_attrs.get("median_best_iteration", 0.0):.1f}')

    seed_cv_means = best_trial.user_attrs.get('seed_cv_means', [])

    if seed_cv_means:
        print('\nSeed results:')

        for seed_index, seed_cv_mean in enumerate(seed_cv_means):
            fold_seed = best_trial.user_attrs[f'seed_{seed_index}_fold_seed']
            seed_cv_std = best_trial.user_attrs[f'seed_{seed_index}_cv_std']
            print(f'Seed {fold_seed}: {seed_cv_mean:.4f} ± {seed_cv_std:.4f}')

    print('\nBest parameters:')

    for param_name, param_value in best_trial.params.items():
        print(f'{param_name}: {param_value}')

    print(f'\nTop {min(top_n, len(trials_df))} trials:')

    seed_mean_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('seed_')
        and column.endswith('_cv_mean')
    ])

    parameter_columns = sorted([
        column
        for column in trials_df.columns
        if column.startswith('params_')
    ])

    columns_to_print = [
        'trial_number',
        'cv_score',
        'seed_cv_std',
        'cv_std',
        *seed_mean_columns,
        *parameter_columns,
    ]

    existing_columns = [
        column
        for column in columns_to_print
        if column in trials_df.columns
    ]

    print(trials_df.head(top_n)[existing_columns].to_string(index=False))