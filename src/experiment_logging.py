import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree               import export_text, plot_tree

from config                     import config
from pathlib                    import Path


def print_section(title: str) -> None:
    '''Print formatted section title.'''

    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


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

    else:
        print('No model specific logging available.')


def print_logistic_regression_info(pipe) -> None:
    '''Print Logistic Regression information.'''

    model = pipe.named_steps['model']

    print(f'\nNumber of iterations: {model.n_iter_[0]}')

    feature_names = (
        pipe.named_steps['preprocessor']
        .get_feature_names_out()
    )

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

    feature_names = (
        pipe.named_steps['preprocessor']
        .get_feature_names_out()
    )

    print_tree_summary(model)
    print_feature_importance(model, feature_names)
    print_tree_structure(model, feature_names)
    save_tree_plot(config, pipe)


def print_tree_summary(model) -> None:
    '''Print Decision Tree summary.'''

    print_section('Tree Summary')

    print(f'Depth: {model.get_depth()}')
    print(f'Leaves: {model.get_n_leaves()}')
    print(f'Nodes: {model.tree_.node_count}')


def print_feature_importance(model, feature_names) -> None:
    '''Print feature importance.'''

    print_section('Feature Importance')

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    })

    importance_df = (
        importance_df
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

    feature_names = (
        pipe.named_steps['preprocessor']
        .get_feature_names_out()
    )

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
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'Tree plot saved: {output_path}')


def log_coefficients(config, pipe, path_to_coefficients) -> None:
    '''Save model coefficients if the model has coefficients.'''

    model = pipe.named_steps['model']

    if not hasattr(model, 'coef_'):
        print('Model has no coefficients. Skipping coefficient logging.')
        return

    feature_names = pipe.named_steps[
        'preprocessor'
    ].get_feature_names_out()

    coef = model.coef_[0]

    print_section('Saving Coefficients')

    coefficients_df = pd.DataFrame({
        'experiment_name': config.general.experiment_name,
        'model_name': config.model.active,
        'feature': feature_names,
        'coef': coef,
        'abs_coef': abs(coef),
        'is_zero': coef == 0,
    }).sort_values('abs_coef', ascending=False)

    path_to_coefficients = Path(path_to_coefficients)

    coefficients_df.to_csv(
        path_to_coefficients,
        mode='a',
        header=not path_to_coefficients.exists(),
        index=False,
    )


def log_experiment_results(config, pipe, cv_score, cv_std, holdout_score, path_to_experiments) -> None:
    '''Save universal experiment results.'''

    active_model = config.model.active
    model = pipe.named_steps['model']
    model_params = dict(config.model.models[active_model])

    experiment_data = {
        'experiment_name': config.general.experiment_name,
        'model_name': active_model,
        'cv_score': cv_score,
        'cv_std': cv_std,
        'holdout_score': holdout_score,
    }

    # Save model hyperparameters as separate columns.
    for param_name, param_value in model_params.items():
        experiment_data[f'model__{param_name}'] = param_value

    experiment_row = pd.DataFrame([experiment_data])

    path_to_experiments = Path(path_to_experiments)

    experiment_row.to_csv(
        path_to_experiments,
        mode='a',
        header=not path_to_experiments.exists(),
        index=False,
    )