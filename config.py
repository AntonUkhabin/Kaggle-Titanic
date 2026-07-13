from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '140_random_forest_max_depth_10',
    },
    'logging': {
        'save_tree_plot': True,
    },
    'paths': {
        'path_to_csv':                  './data/train.csv',
        'path_to_kaggle_test':          './data/test.csv',
        'path_to_submission':           './outputs/submission.csv',
        'path_to_experiments':          './outputs/experiments.csv',
        'path_to_coefficients':         './outputs/coefficients.csv',
        'path_to_logs':                 './outputs/logs',
        'path_to_tree_plots':           './outputs/tree_plots',

        'path_to_optuna_trials':        './outputs/optuna/135_decision_tree_optuna_100_trials.csv',
        'path_to_optuna_best_params':   './outputs/optuna/135_decision_tree_optuna_100_trials_best_params.yaml',
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
    'optuna': {
        'study_name': 'decision_tree_tuning',
        'direction': 'maximize',
        'n_trials': 100,
        'show_progress_bar': True,

        'search_space': {
            'criterion': [
                'gini',
                'entropy',
                'log_loss',
            ],
            'max_depth': {
                'low': 3,
                'high': 12,
            },
            'min_samples_split': {
                'low': 2,
                'high': 30,
            },
            'min_samples_leaf': {
                'low': 1,
                'high': 15,
            },
            'max_leaf_nodes': [
                None,
                10,
                15,
                18,
                20,
                22,
                25,
                30,
                35,
                40,
            ],
            'ccp_alpha': [
                0.0,
                0.0005,
                0.001,
                0.0015,
                0.002,
                0.003,
                0.004,
                0.005,
                0.01,
            ],
        },
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
        'active': 'random_forest',

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
                'criterion': 'log_loss',
                'max_depth': 11,
                'min_samples_split': 10,
                'min_samples_leaf': 3,
                'max_leaf_nodes': 18,
                'ccp_alpha': 0,
                'class_weight': None,
                'random_state': 0xC0FFEE,
            },

            'random_forest': {
                'n_estimators': 300,
                'criterion': 'gini',
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': None,
                'random_state': 0xC0FFEE,
                'n_jobs': -1,
                'oob_score': True,
            },
        },
    },
}

config = OmegaConf.create(config)