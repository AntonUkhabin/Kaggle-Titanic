from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '83_decision_tree_prepruning_baseline',
    },
    'logging': {
        'save_tree_plot': True,
    },
    'paths': {
        'path_to_csv':              './data/train.csv',
        'path_to_kaggle_test':      './data/test.csv',
        'path_to_submission':       './outputs/submission.csv',
        'path_to_experiments':      './outputs/experiments.csv',
        'path_to_coefficients':     './outputs/coefficients.csv',
        'path_to_logs':             './outputs/logs',
        'path_to_tree_plots':       './outputs/tree_plots',
    },
    'training': {
        'num_epochs': 200,
        'early_stopping_epochs': 20,
        'lr': 0.1,
    },
    'dataloader_params': {
        'shuffle': True,
    },
    'split': {
        'n_splits': 5,
        'test_size': 0.2,
    },
    'loss': {
        'name': 'log_loss',
        'params': {
        },  
    },
    'metric': {
        'name': 'accuracy_score',
        'params': {
        },
    },
    'model': {
        'active': 'decision_tree',

        'models': {
            'logistic_regression': {
                'solver': 'saga',
                'l1_ratio': 0.5,
                'C': 0.7,
                'max_iter': 1000,
                'random_state': 0xC0FFEE,
            },
            
            'knn': { # Remove Embarked, IsAlone
                'n_neighbors': 6,
                'weights': 'uniform',
                'metric': 'minkowski',
                'p': 2,
            },

            'decision_tree': {
                'criterion': 'gini',
                'max_depth': 3,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_leaf_nodes': None,
                'class_weight': None,
                'random_state': 0xC0FFEE,
            },
        },
    },
}

config = OmegaConf.create(config)