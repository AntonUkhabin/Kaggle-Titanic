from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree


def print_section(title: str) -> None:
    '''Print formatted section title.'''

    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def print_classification_metrics(name, metrics):
    '''Print classification and probability metrics.'''

    print(f'{name} Accuracy: {metrics["accuracy"]:.4f}')
    print(f'{name} Precision: {metrics["precision"]:.4f}')
    print(f'{name} Recall: {metrics["recall"]:.4f}')
    print(f'{name} F1: {metrics["f1"]:.4f}')
    print(f'{name} ROC-AUC: {metrics["roc_auc"]:.4f}')
    print(f'{name} PR-AUC: {metrics["pr_auc"]:.4f}')
    print(f'{name} Logloss: {metrics["logloss"]:.4f}')
    print(f'{name} Brier: {metrics["brier"]:.4f}')
    print(f'{name} Confusion Matrix: TN={metrics["tn"]}, FP={metrics["fp"]}, FN={metrics["fn"]}, TP={metrics["tp"]}')


def print_model_info(pipe, config) -> None:
    '''Print fitted model information.'''

    active_model = config.model.active

    print_section('Model Information')

    print(f'Active model: {active_model}')
    print(pipe.named_steps['model'])

    if active_model == 'logistic_regression':
        print_logistic_regression_info(pipe)

    elif active_model == 'knn':
        print_knn_info(pipe)

    elif active_model == 'decision_tree':
        print_decision_tree_info(pipe, config)
    
    elif active_model == 'random_forest':
        print_random_forest_info(pipe)

    else:
        print('No model specific logging available.')


def print_logistic_regression_info(pipe) -> None:
    '''Print Logistic Regression information.'''

    model = pipe.named_steps['model']

    print(f'\nNumber of iterations: {model.n_iter_[0]}')

    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    coef = model.coef_[0]

    total_coef = len(coef)
    zero_coef = (coef == 0).sum()
    non_zero_coef = (coef != 0).sum()

    print(f'Total coefficients: {total_coef}')
    print(f'Zero coefficients: {zero_coef}')
    print(f'Non-zero coefficients: {non_zero_coef}')

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coef,
        'abs_coef': abs(coef),
    }).sort_values('abs_coef', ascending=False)

    print(coef_df.to_string(index=False))


def print_knn_info(pipe) -> None:
    '''Print KNN information.'''

    model = pipe.named_steps['model']

    print(f'n_neighbors: {model.n_neighbors}')
    print(f'weights: {model.weights}')
    print(f'metric: {model.metric}')
    print(f'p: {model.p}')


def print_decision_tree_info(pipe, config) -> None:
    '''Print Decision Tree information and save optional tree plot.'''

    model = pipe.named_steps['model']

    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    print_tree_summary(model)
    print_feature_importance(model, feature_names)
    print_tree_structure(model, feature_names)
    save_tree_plot(config, pipe)


def print_random_forest_info(pipe) -> None:
    '''Print Random Forest information and feature importance.'''

    model = pipe.named_steps['model']

    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    tree_depths = [estimator.get_depth() for estimator in model.estimators_]
    tree_leaves = [estimator.get_n_leaves() for estimator in model.estimators_]

    print_section('Random Forest Summary')

    print(f'Number of trees: {model.n_estimators}')
    print(f'Criterion: {model.criterion}')
    print(f'Max depth: {model.max_depth}')
    print(f'Min samples split: {model.min_samples_split}')
    print(f'Min samples leaf: {model.min_samples_leaf}')
    print(f'Max features: {model.max_features}')
    print(f'Bootstrap: {model.bootstrap}')
    
    if model.oob_score:
        print(f'OOB Accuracy: {model.oob_score_:.4f}')
    
    print(f'Mean tree depth: {np.mean(tree_depths):.2f}')
    print(f'Min tree depth: {np.min(tree_depths)}')
    print(f'Max tree depth: {np.max(tree_depths)}')
    print(f'Mean leaves per tree: {np.mean(tree_leaves):.2f}')

    print_feature_importance(
        model=model,
        feature_names=feature_names,
    )


def print_tree_summary(model) -> None:
    '''Print Decision Tree summary.'''

    print_section('Tree Summary')

    print(f'Criterion: {model.criterion}')
    print(f'Max depth: {model.max_depth}')
    print(f'Depth: {model.get_depth()}')
    print(f'Leaves: {model.get_n_leaves()}')
    print(f'Nodes: {model.tree_.node_count}')


def print_feature_importance(model, feature_names) -> None:
    '''Print feature importance.'''

    print_section('Feature Importance')

    importance_df = (
        pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
        })
        .query('importance > 0')
        .assign(importance=lambda data: data['importance'].round(4))
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )

    print(importance_df.to_string(index=False))


def print_tree_structure(model, feature_names) -> None:
    '''Print tree structure.'''

    print_section('Tree Structure')

    tree_text = export_text(
        model,
        feature_names=list(feature_names),
    )

    print(tree_text)


def save_tree_plot(config, pipe) -> None:
    '''Save Decision Tree visualization as PNG.'''

    if config.model.active != 'decision_tree':
        return

    if not config.logging.save_tree_plot:
        return

    model = pipe.named_steps['model']

    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    output_dir = Path(config.paths.path_to_tree_plots)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{config.general.experiment_name}.png'

    plt.figure(figsize=(24, 12))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['Died', 'Survived'],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Tree plot saved: {output_path}')


def save_oof_predictions(train_cv_df, oof_probabilities, experiment_name, output_dir) -> None:
    '''Save detailed out-of-fold predictions and errors to CSV.'''

    oof_predictions = (oof_probabilities > 0.5).astype(int)

    columns_to_save = [
        'PassengerId',
        'Survived',
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'SibSp',
        'Parch',
        'Embarked',
        'Name',
        'Cabin',
        'Ticket'
    ]

    oof_df = train_cv_df[columns_to_save].copy()

    oof_df['oof_probability'] = oof_probabilities
    oof_df['oof_prediction'] = oof_predictions
    oof_df['is_correct'] = oof_df['Survived'] == oof_df['oof_prediction']

    oof_df['prediction_type'] = np.select(
        [
            (oof_df['Survived'] == 1) & (oof_df['oof_prediction'] == 1),
            (oof_df['Survived'] == 0) & (oof_df['oof_prediction'] == 0),
            (oof_df['Survived'] == 1) & (oof_df['oof_prediction'] == 0),
            (oof_df['Survived'] == 0) & (oof_df['oof_prediction'] == 1),
        ],
        [
            'true_positive',
            'true_negative',
            'false_negative',
            'false_positive',
        ],
        default='unknown',
    )

    oof_df['confidence'] = np.where(
        oof_df['oof_prediction'] == 1,
        oof_df['oof_probability'],
        1 - oof_df['oof_probability'],
    )

    oof_df['distance_to_threshold'] = (
        oof_df['oof_probability'] - 0.5
    ).abs()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_output_path = output_dir / f'{experiment_name}.csv'
    errors_output_path = output_dir / f'{experiment_name}_errors.csv'

    oof_df.to_csv(oof_output_path, index=False, sep=';', encoding='utf-8-sig')

    errors_df = (
        oof_df
        .query('is_correct == False')
        .sort_values('confidence', ascending=False)
    )

    errors_df.to_csv(errors_output_path, index=False, sep=';', encoding='utf-8-sig')

    print(f'OOF predictions saved: {oof_output_path}')
    print(f'OOF errors saved: {errors_output_path}')


def save_torch_training_plots(histories, best_epochs, experiment_name, output_dir):
    '''Save loss and accuracy curves for every PyTorch fold.'''

    if not histories:
        return None

    n_folds = len(histories)
    figure, axes = plt.subplots(n_folds, 2, figsize=(14, 4 * n_folds), squeeze=False)

    for fold, (history, best_epoch) in enumerate(zip(histories, best_epochs)):
        epochs = np.arange(1, len(history['train_loss']) + 1)

        loss_axis = axes[fold, 0]
        loss_axis.plot(epochs, history['train_loss'], label='Train loss')
        loss_axis.plot(epochs, history['val_loss'], label='Validation loss')
        loss_axis.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')
        loss_axis.set_title(f'Fold {fold} — Loss')
        loss_axis.set_xlabel('Epoch')
        loss_axis.set_ylabel('Loss')
        loss_axis.set_ylim(bottom=0)
        loss_axis.grid(alpha=0.3)
        loss_axis.legend()

        accuracy_axis = axes[fold, 1]
        accuracy_axis.plot(epochs, history['train_accuracy'], label='Train accuracy')
        accuracy_axis.plot(epochs, history['val_accuracy'], label='Validation accuracy')
        accuracy_axis.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')
        accuracy_axis.set_title(f'Fold {fold} — Accuracy')
        accuracy_axis.set_xlabel('Epoch')
        accuracy_axis.set_ylabel('Accuracy')
        accuracy_axis.set_ylim(0, 1)
        accuracy_axis.grid(alpha=0.3)
        accuracy_axis.legend()

    figure.suptitle(f'PyTorch training history — {experiment_name}', fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{experiment_name}.png'

    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

    print(f'PyTorch training plots saved: {output_path}')

    return output_path