import numpy as np
import sys

from sklearn.model_selection    import train_test_split

from config                     import config
from src.data                   import load_data
from src.utils                  import setup_run_logging
from src.train_functions        import (build_pipeline,
                                        create_submission,
                                        cross_validate_model_with_early_stopping,
                                        cross_validate_standard,
                                        evaluate_model,
                                        predict_with_catboost_ensemble,
                                        predict_with_pipeline)
from src.experiment_logging     import print_section, print_model_info, log_experiment_results, log_coefficients, save_oof_predictions


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

        print_section('Cross Validation')

        training_strategy = config.training.strategy_by_model[config.model.active]

        if training_strategy == 'standard_cv_full_fit':
            scores, pipe, oof_probabilities = cross_validate_standard(
                train_cv_df=train_cv_df,
                target_col='Survived',
                config=config,
            )

            best_iterations = None
            fold_models = []
            ensemble_size = 1

        elif training_strategy == 'early_stopping_cv_fold_ensemble':
            scores, best_iterations, fold_models, oof_probabilities = cross_validate_model_with_early_stopping(
                train_cv_df=train_cv_df,
                target_col='Survived',
                config=config,
            )

            ensemble_size = len(fold_models)

        else:
            raise ValueError(f'Unknown training strategy: {training_strategy}')

        print_model_info(pipe, config)

        # Calculate mean and standard deviation of CV scores.
        cv_score = sum(scores) / len(scores)
        cv_std = np.std(scores)
        oof_predictions = (oof_probabilities > 0.5).astype(int)
        oof_score = evaluate_model(train_cv_df['Survived'], oof_predictions)

        print(f'\nMean CV Accuracy: {cv_score:.4f}')
        print(f'CV STD: {cv_std:.4f}')
        print(f'OOF Accuracy: {oof_score:.4f}')

        save_oof_predictions(
            train_cv_df=train_cv_df,
            oof_probabilities=oof_probabilities,
            experiment_name=config.general.experiment_name,
            output_dir=config.paths.path_to_oof,
        )

        if best_iterations is not None:
            print(f'Best iterations: {best_iterations}')
            print(f'Median best iteration: {int(np.median(best_iterations))}')

        print(f'Number of prediction models: {ensemble_size}')

        # Split holdout_df into features and target.
        features_holdout = holdout_df.drop(columns=['Survived'])
        labels_holdout = holdout_df['Survived']

        # Generate predictions for holdout_df.
        print_section('Holdout Evaluation')

        if training_strategy == 'standard_cv_full_fit':
            holdout_preds, holdout_probabilities = predict_with_pipeline(features_holdout, pipe)
        else:
            holdout_preds, holdout_probabilities = predict_with_catboost_ensemble(features_holdout, fold_models)

        holdout_score = evaluate_model(labels_holdout, holdout_preds)

        print(f'Holdout Accuracy: {holdout_score:.4f}')

        print_section('Experiment Logging')
        
        log_experiment_results(
            config=config,
            pipe=pipe,
            cv_score=cv_score,
            cv_std=cv_std,
            holdout_score=holdout_score,
            path_to_experiments=config.paths.path_to_experiments,
            ensemble_size=ensemble_size,
            best_iterations=best_iterations,
        )
        
        print('Experiment results saved.')

        log_coefficients(
            config=config,
            pipe=pipe,
            path_to_coefficients=config.paths.path_to_coefficients,
        )

        # Generate predictions for Kaggle test.csv.
        print_section('Kaggle Submission')

        if training_strategy == 'standard_cv_full_fit':
            kaggle_test_preds, kaggle_test_probabilities = predict_with_pipeline(kaggle_test_df, pipe)
        else:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_catboost_ensemble(kaggle_test_df, fold_models)

        print(f'Number of prediction models: {ensemble_size}')
        print(f'Mean predicted survival probability: {np.mean(kaggle_test_probabilities):.4f}')

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