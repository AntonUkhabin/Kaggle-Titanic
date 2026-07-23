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
                                        predict_with_early_stopping_ensemble,
                                        predict_with_pipeline,
                                        calculate_classification_metrics,
                                        predict_with_pipeline_ensemble,
                                        )
from src.experiment_logging     import (log_coefficients,
                                        log_experiment_results,
                                        print_classification_metrics,
                                        print_model_info,
                                        print_section,
                                        save_oof_predictions,
                                        )


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

        training_profile = config.training.model_profiles[config.model.active]
        use_early_stopping = training_profile.early_stopping
        use_fold_ensemble = training_profile.fold_ensemble

        if use_early_stopping and not use_fold_ensemble:
            raise ValueError('Early stopping currently requires fold ensemble.')

        if use_early_stopping:
            scores, best_iterations, fold_models, oof_probabilities = cross_validate_model_with_early_stopping(
                train_cv_df=train_cv_df,
                target_col='Survived',
                config=config,
            )

        else:
            scores, pipe, fold_models, oof_probabilities = cross_validate_standard(
                train_cv_df=train_cv_df,
                target_col='Survived',
                config=config,
                keep_fold_models=use_fold_ensemble,
            )

            best_iterations = None
        
        ensemble_size = len(fold_models) if use_fold_ensemble else 1

        print_model_info(pipe, config)

        # Calculate mean and standard deviation of CV scores.
        cv_score = sum(scores) / len(scores)
        cv_std = np.std(scores)
        oof_predictions = (oof_probabilities > 0.5).astype(int)

        oof_metrics = calculate_classification_metrics(
            train_cv_df['Survived'],
            oof_predictions,
            oof_probabilities,
        )

        print(f'\nMean CV Accuracy: {cv_score:.4f}')
        print(f'CV STD: {cv_std:.4f}')
        print_classification_metrics('OOF', oof_metrics)

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

        if not use_fold_ensemble:
            holdout_preds, holdout_probabilities = predict_with_pipeline(
                features_holdout,
                pipe,
            )

        elif use_early_stopping:
            holdout_preds, holdout_probabilities = predict_with_early_stopping_ensemble(
                features_holdout,
                fold_models,
            )

        else:
            holdout_preds, holdout_probabilities = predict_with_pipeline_ensemble(
                features_holdout,
                fold_models,
            )

        holdout_metrics = calculate_classification_metrics(labels_holdout, holdout_preds, holdout_probabilities)
        holdout_score = holdout_metrics['accuracy']

        print_classification_metrics('Holdout', holdout_metrics)

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

        if not use_fold_ensemble:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_pipeline(
                kaggle_test_df,
                pipe,
            )

        elif use_early_stopping:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_early_stopping_ensemble(
                kaggle_test_df,
                fold_models,
            )

        else:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_pipeline_ensemble(
                kaggle_test_df,
                fold_models,
            )

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