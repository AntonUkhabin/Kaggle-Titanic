import sys

from sklearn.model_selection import train_test_split

from config import config
from src.data import load_data
from src.experiment_logging import print_section
from src.optuna_tuning import (
    print_optuna_results,
    run_optuna_study,
    save_best_params,
    save_optuna_trials,
)
from src.train_functions import build_pipeline
from src.utils import setup_run_logging


def main() -> None:
    '''Run Optuna tuning for the active model.'''

    logger = setup_run_logging(config)

    try:
        print_section(f'Experiment: {config.general.experiment_name}')

        # Load train.csv and Kaggle test.csv.
        train_df, _ = load_data(config)

        # Use exactly the same split as the regular training workflow.
        # Holdout is created for consistency, but it is not used by Optuna.
        train_cv_df, _ = train_test_split(
            train_df,
            test_size=config.split.test_size,
            random_state=config.general.seed,
            shuffle=config.dataloader_params.shuffle,
            stratify=train_df['Survived'],
        )

        # Build the unchanged full baseline pipeline.
        base_pipe = build_pipeline(config)

        print_section('Optuna Study')

        print(f'Study name: {config.optuna.study_name}')
        print(f'Direction: {config.optuna.direction}')
        print(f'Number of trials: {config.optuna.n_trials}')
        print(f'CV folds: {config.split.n_splits}')
        print('Holdout evaluation: disabled during tuning')

        study = run_optuna_study(
            train_cv_df=train_cv_df,
            target_col='Survived',
            base_pipe=base_pipe,
            config=config,
        )

        print_section('Saving Optuna Results')

        trials_df = save_optuna_trials(
            study=study,
            output_path=(
                config.paths.path_to_optuna_trials
            ),
        )

        save_best_params(
            study=study,
            output_path=(
                config.paths.path_to_optuna_best_params
            ),
        )

        print(
            'Trials saved to: '
            f'{config.paths.path_to_optuna_trials}'
        )
        print(
            'Best parameters saved to: '
            f'{config.paths.path_to_optuna_best_params}'
        )

        print_section('Optuna Results')

        print_optuna_results(
            study=study,
            trials_df=trials_df,
            top_n=10,
        )

    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()