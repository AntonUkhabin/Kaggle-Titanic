import numpy as np
import sys

from sklearn.model_selection    import train_test_split

from config                     import config
from src.data                   import load_data
from src.utils                  import setup_run_logging
from src.train_functions        import evaluate_model, create_submission, cross_validate_model, build_pipeline, cross_validate_model_with_early_stopping, predict_with_catboost_ensemble
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

        # Run cross-validation on train_cv_df.
        print_section('Cross Validation')
        
        scores, best_iterations, fold_models, oof_probabilities = cross_validate_model_with_early_stopping(
            train_cv_df=train_cv_df,
            target_col='Survived',
            config=config,
        )

        # Calculate mean and standard deviation of CV scores.
        cv_score = sum(scores) / len(scores)
        cv_std = np.std(scores)
        oof_predictions = (oof_probabilities >= 0.5).astype(int)
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

        print('Best iterations: ' f'{best_iterations}')
        print('Median best iteration: ' f'{int(np.median(best_iterations))}')
        print(f'Number of fold models: {len(fold_models)}')

        # Split holdout_df into features and target.
        features_holdout = holdout_df.drop(columns=['Survived'])
        labels_holdout = holdout_df['Survived']

        # Generate predictions for holdout_df.
        print_section('Holdout Evaluation')

        ensemble_holdout_preds, ensemble_holdout_probabilities = predict_with_catboost_ensemble(
            features=features_holdout,
            fold_models=fold_models,
        )

        ensemble_holdout_score = evaluate_model(
            labels_true=labels_holdout,
            labels_pred=ensemble_holdout_preds,
        )

        print(f'Ensemble Holdout Accuracy: {ensemble_holdout_score:.4f}')

        print_section('Experiment Logging')
        
        log_experiment_results(
            config=config,
            pipe=pipe,
            cv_score=cv_score,
            cv_std=cv_std,
            holdout_score=ensemble_holdout_score,
            path_to_experiments=config.paths.path_to_experiments,
            ensemble_size=len(fold_models),
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
        # kaggle_test_preds = pipe.predict(kaggle_test_df)
        kaggle_test_preds, kaggle_test_probabilities = predict_with_catboost_ensemble(
            features=kaggle_test_df,
            fold_models=fold_models,
        )

        print(f'Number of ensemble models: {len(fold_models)}')
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