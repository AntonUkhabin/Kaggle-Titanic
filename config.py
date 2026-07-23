from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '219_refactoring_test',
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
        'path_to_oof':                  './outputs/oof',
        'path_to_tree_plots':           './outputs/tree_plots',

        'path_to_optuna_trials':        './outputs/optuna/199_catboost_optuna.csv',
        'path_to_optuna_best_params':   './outputs/optuna/199_catboost_optuna_best_params.yaml',
    },
    'training': {
        'early_stopping_rounds': 100,
        'fold_seed': 0xC0FFEE,

        'model_profiles': {
            'logistic_regression': {
                'early_stopping': False,
                'fold_ensemble': False,
            },
            'knn': {
                'early_stopping': False,
                'fold_ensemble': True,
            },
            'decision_tree': {
                'early_stopping': False,
                'fold_ensemble': False,
            },
            'random_forest': {
                'early_stopping': False,
                'fold_ensemble': True,
            },
            'catboost': {
                'early_stopping': True,
                'fold_ensemble': True,
            },
            'lightgbm': {
                'early_stopping': True,
                'fold_ensemble': True,
            },
        },
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
        'active': 'catboost',

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
                'max_depth': 13,
                'min_samples_split': 5,
                'min_samples_leaf': 3,
                'max_features': 'log2',
                'bootstrap': True,
                'oob_score': True,
                'class_weight': None,
                'random_state': 0xC0FFEE,
                'n_jobs': -1,
            },

            'catboost': {
                'iterations': 2000,
                'learning_rate': 0.1,
                'depth': 5,
                'l2_leaf_reg': 3.12,
                'loss_function': 'Logloss',
                'eval_metric': 'Accuracy',
                'custom_metric': ['Logloss'],
                'nan_mode': 'Min',
                'random_seed': 0xC0FFEE,
                'thread_count': 8,
                'verbose': 0,
                'allow_writing_files': False,
                'random_strength': 0.24,
                'bootstrap_type': 'MVS',
            },

            'lightgbm': {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_error',

                'n_estimators': 2000,
                'learning_rate': 0.03,

                'num_leaves': 15,
                'max_depth': 4,
                'min_child_samples': 10,
                'min_child_weight': 0.001,
                'min_split_gain': 0.0,

                'subsample': 1.0,
                'subsample_freq': 0,
                'colsample_bytree': 1.0,

                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'class_weight': None,

                'cat_smooth': 10.0,
                'cat_l2': 10.0,
                'min_data_per_group': 100,
                'max_cat_threshold': 32,
                'max_cat_to_onehot': 4,

                'random_state': 0xC0FFEE,
                'n_jobs': 15,
                'deterministic': True,
                'force_row_wise': True,
                'verbosity': -1,
            },
        },
    },

    'optuna': {
        'study_name': 'catboost_tuning',
        'direction': 'maximize',
        'n_trials': 100,
        'show_progress_bar': True,

        'search_spaces': {

        'catboost': {
            'depth': {
                'low': 4,
                'high': 9,
                },
            'learning_rate': {
                'low': 0.01,
                'high': 0.10,
                },
            'l2_leaf_reg': {
                'low': 1.0,
                'high': 10.0,
                },
            'random_strength': {
                'low': 0.0,
                'high': 2.0,
                },
            },
            
            'decision_tree': {
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

            'random_forest': {
                'criterion': [
                    'gini',
                    'entropy',
                    'log_loss',
                ],

                'n_estimators': {
                    'low': 200,
                    'high': 700,
                    'step': 50,
                },

                'max_depth': {
                    'low': 6,
                    'high': 14,
                },

                'min_samples_split': {
                    'low': 2,
                    'high': 20,
                },

                'min_samples_leaf': {
                    'low': 2,
                    'high': 8,
                },

                'max_features': [
                    'sqrt',
                    'log2',
                    0.5,
                    0.75,
                ],
            },
        },
    }, 
}

config = OmegaConf.create(config)