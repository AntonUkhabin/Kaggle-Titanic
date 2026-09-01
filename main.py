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
from src.experiment_logging     import (print_classification_metrics,
                                        print_model_info,
                                        print_section,
                                        save_oof_predictions,
                                        save_torch_training_plots,
                                        )
from src.torch_training         import cross_validate_neural_network, predict_with_neural_network_ensemble                                        


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
        is_torch_model = config.model.active == 'dnn'
        pipe = None if is_torch_model else build_pipeline(config)

        print_section(f'Experiment: {config.general.experiment_name}')

        print_section('Cross Validation')

        if is_torch_model:
            # DNN uses early stopping within each fold and averages fold predictions.
            scores, best_epochs, fold_models, oof_probabilities, histories = cross_validate_neural_network(train_cv_df, 'Survived', config)

            best_iterations = best_epochs
            use_early_stopping = True
            use_fold_ensemble = True

        elif config.model.active in ('catboost', 'lightgbm', 'xgboost'):
            # Each boosting fold selects its own tree count using early stopping.
            # Keep all fold models for ensemble prediction.
            scores, best_iterations, fold_models, oof_probabilities = cross_validate_model_with_early_stopping(train_cv_df=train_cv_df, target_col='Survived', config=config)

            use_early_stopping = True
            use_fold_ensemble = True
            histories = None

        else:
            # Standard models either retain fold models or refit on all Train/CV data.
            use_early_stopping = False
            use_fold_ensemble = config.training.fold_ensemble[config.model.active]

            scores, pipe, fold_models, oof_probabilities = cross_validate_standard(train_cv_df=train_cv_df, target_col='Survived', config=config, keep_fold_models=use_fold_ensemble)

            best_iterations = None
            histories = None
        
        ensemble_size = len(fold_models) if use_fold_ensemble else 1

        if is_torch_model and config.logging.save_torch_plots:
            save_torch_training_plots(histories, best_epochs, config.general.experiment_name, config.paths.path_to_torch_plots)
        
        if is_torch_model or use_early_stopping:
            print_section('Model Information')
            print(f'Active model: {config.model.active}')
            print('Model from fold 0:')
            print(fold_models[0][1])
        else:
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
            iteration_name = 'epochs' if is_torch_model else 'iterations'
            print(f'Best {iteration_name}: {best_iterations}')
            print(f'Median best {iteration_name[:-1]}: {int(np.median(best_iterations))}')

        print(f'Number of prediction models: {ensemble_size}')

        # Split holdout_df into features and target.
        features_holdout = holdout_df.drop(columns=['Survived'])
        labels_holdout = holdout_df['Survived']

        # Generate predictions for holdout_df.
        print_section('Holdout Evaluation')

        if is_torch_model:
            holdout_preds, holdout_probabilities = predict_with_neural_network_ensemble(features_holdout, fold_models, config)

        elif not use_fold_ensemble:
            holdout_preds, holdout_probabilities = predict_with_pipeline(features_holdout, pipe)

        elif use_early_stopping:
            holdout_preds, holdout_probabilities = predict_with_early_stopping_ensemble(features_holdout, fold_models)

        else:
            holdout_preds, holdout_probabilities = predict_with_pipeline_ensemble(features_holdout, fold_models)

        holdout_metrics = calculate_classification_metrics(labels_holdout, holdout_preds, holdout_probabilities)

        print_classification_metrics('Holdout', holdout_metrics)

        
        # Generate predictions for Kaggle test.csv.
        print_section('Kaggle Submission')

        if is_torch_model:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_neural_network_ensemble(kaggle_test_df, fold_models, config)

        elif not use_fold_ensemble:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_pipeline(kaggle_test_df, pipe)

        elif use_early_stopping:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_early_stopping_ensemble(kaggle_test_df, fold_models)

        else:
            kaggle_test_preds, kaggle_test_probabilities = predict_with_pipeline_ensemble(kaggle_test_df, fold_models)

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
        sys.stderr = logger.original_stderr
        logger.close()


if __name__ == "__main__":
    main()