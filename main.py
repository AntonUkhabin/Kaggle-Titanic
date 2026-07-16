import numpy as np
import sys

from sklearn.model_selection    import train_test_split

from config                     import config
from src.data                   import load_data
from src.utils                  import setup_run_logging
from src.train_functions        import evaluate_model, create_submission, cross_validate_model, build_pipeline, cross_validate_model_with_early_stopping
from src.experiment_logging     import print_section, print_model_info, log_experiment_results, log_coefficients


def main() -> None:
    '''Run the full Titanic machine learning pipeline.'''

    logger = setup_run_logging(config)

    try:

        # Load train.csv and Kaggle test.csv
        train_df, kaggle_test_df = load_data(config)

        # Split train.csv into:
        # - train_cv_df: data for cross-validation
        # - holdout_df: final local test, used only after CV
        train_cv_df, holdout_df = train_test_split(
            train_df,
            test_size=config.split.test_size,
            random_state=config.general.seed,
            shuffle=config.dataloader_params.shuffle,
            stratify=train_df["Survived"]
        )

        # Build full pipeline: feature engineering + preprocessing + model.
        pipe = build_pipeline(config)

        print_section(f'Experiment: {config.general.experiment_name}')

        # Run cross-validation on train_cv_df.
        print_section('Cross Validation')
        scores, best_iterations = (
            cross_validate_model_with_early_stopping(
                train_cv_df=train_cv_df,
                target_col='Survived',
                config=config,
            )
        )

        # Calculate mean and standard deviation of CV scores.
        cv_score = sum(scores) / len(scores)
        cv_std = np.std(scores)

        print(f'\nMean CV Accuracy: {cv_score:.4f}')
        print(f'CV STD: {cv_std:.4f}')

        print('Best iterations: ' f'{best_iterations}')
        print('Median best iteration: ' f'{int(np.median(best_iterations))}')

        final_iterations = int(np.median(best_iterations))
        pipe.set_params(model__iterations=final_iterations)

        print(f'Final number of iterations: {final_iterations}')

        # Split train_cv_df into features and target.
        features_train_cv = train_cv_df.drop(columns=['Survived'])
        labels_train_cv = train_cv_df['Survived']

        # Fit pipeline on the full train_cv_df.
        print_section('Model Training')
        pipe.fit(features_train_cv, labels_train_cv)
        print_model_info(pipe, config)

        # Split holdout_df into features and target.
        features_holdout = holdout_df.drop(columns=['Survived'])
        labels_holdout = holdout_df['Survived']

        # Generate predictions for holdout_df.
        print_section('Holdout Evaluation')
        holdout_preds = pipe.predict(features_holdout)

        # Calculate holdout accuracy.
        holdout_score = evaluate_model(
            labels_true=labels_holdout,
            labels_pred=holdout_preds,
        )

        print(f'\nHoldout Accuracy: {holdout_score:.4f}')

        print_section('Experiment Logging')
        log_experiment_results(
            config=config,
            pipe=pipe,
            cv_score=cv_score,
            cv_std=cv_std,
            holdout_score=holdout_score,
            path_to_experiments=config.paths.path_to_experiments,
        )
        print('Experiment results saved.')

        log_coefficients(
            config=config,
            pipe=pipe,
            path_to_coefficients=config.paths.path_to_coefficients,
        )

        # Generate predictions for Kaggle test.csv.
        print_section('Kaggle Submission')
        kaggle_test_preds = pipe.predict(kaggle_test_df)

        # Create submission.csv for Kaggle.
        create_submission(
            test_df=kaggle_test_df,
            test_preds=kaggle_test_preds,
            path_to_submission=config.paths.path_to_submission,
        )

        print('\nSubmission file saved successfully.')

    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()