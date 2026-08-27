from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '309_dnn_64_32_16_lr_0_001',
    },
    'logging': {
        'save_tree_plot': True,
        'save_torch_plots': True,
        'torch_log_interval': 1, # Уменьшить количество выводимых эпох в терминале. 10 - выводить каждые 10 эпох
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
        'path_to_checkpoints':          './outputs/checkpoints',
        'path_to_torch_plots':          './outputs/torch_plots',

        'path_to_optuna_trials':        './outputs/optuna/292_xgboost_optuna_3_seeds_100_trials.csv',
        'path_to_optuna_best_params':   './outputs/optuna/292_xgboost_optuna_3_seeds_100_trials_best_params.yaml',
    },
    'training': {
        'early_stopping_rounds': 500, # Classic models
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
            'xgboost': {
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
        'active': 'dnn',

        'models': {

            'dnn': {
                'epochs': 200,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_rounds': 15,
                'min_delta': 1e-4,
                'num_workers': 0,
            },

            'logistic_regression': {
                'solver': 'saga',
                'l1_ratio': 0.5,
                'C': 0.7,
                'max_iter': 1000,
                'random_state': 0xC0FFEE,
            },
            
            'knn': {
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
                # Core boosting parameters.
                'n_estimators': 2000,
                'learning_rate': 0.1,
                'objective': 'binary',
                'metric': 'binary_error',

                # Tree complexity.
                'num_leaves': 7,
                'max_depth': 4,
                'max_bin': 255,

                # Leaf and split regularization.
                'min_child_samples': 20,
                'min_child_weight': 1.0,
                'min_split_gain': 0.0,

                # Row and feature sampling.
                'subsample': 1.0,
                'subsample_freq': 0,
                'colsample_bytree': 1.0,

                # L1 and L2 regularization.
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,

                # Categorical feature handling.
                'cat_smooth': 10.0,
                'cat_l2': 10.0,
                'min_data_per_group': 100,
                'max_cat_threshold': 32,
                'max_cat_to_onehot': 4,

                # Reproducibility and performance.
                'random_state': 0xC0FFEE,
                'n_jobs': 15,
                'verbosity': -1,
                'deterministic': True,
                'force_row_wise': True,
            },
            'xgboost': {
                # Core boosting parameters.
                'n_estimators': 2000,
                'learning_rate': 0.11176949123089937,
                'objective': 'binary:logistic',
                'eval_metric': 'error',

                # Tree complexity.
                'max_depth': 2,
                'min_child_weight': 1.0,
                'gamma': 0.0,

                # Row and feature sampling.
                'subsample': 0.8364402681078542,
                'colsample_bytree': 0.8244924870903084,

                # L1 and L2 regularization.
                'reg_alpha': 0.09960972038669126,
                'reg_lambda': 0.5828846693956847,

                # Tree construction.
                'tree_method': 'hist',
                'enable_categorical': False,

                # Reproducibility and performance.
                'random_state': 0xC0FFEE,
                'n_jobs': 15,
                'verbosity': 0,
            },
        },
    },

    'optuna': {
        'study_name': 'xgboost_tuning_3_seeds',
        'direction': 'maximize',
        'n_trials': 100,
        'show_progress_bar': True,

        'fold_seeds': [
            0xC0FFEE,
            42,
            2026,
        ],

        'enqueued_trials': {
            
            'xgboost': [
                {
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 0.9,
                    'colsample_bytree': 1.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 1.0,
                },
            ],
            
            'lightgbm': [
                {
                    'learning_rate': 0.1,
                    'num_leaves': 7,
                    'max_depth': 4,
                    'min_child_samples': 20,
                    'min_child_weight': 1.0,
                    'min_split_gain': 0.0,
                    'subsample_freq': 0,
                    'colsample_bytree': 1.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.0,
                    'cat_smooth': 10.0,
                    'cat_l2': 10.0,
                    'min_data_per_group': 100,
                    'max_cat_threshold': 32,
                    'max_cat_to_onehot': 4,
                    'max_bin': 255,
                },
                {
                    'learning_rate': 0.11784745281602944,
                    'num_leaves': 4,
                    'max_depth': 7,
                    'min_child_samples': 10,
                    'min_child_weight': 0.43511604559295053,
                    'min_split_gain': 0.08776787070217627,
                    'subsample_freq': 1,
                    'subsample': 0.9696707640811784,
                    'colsample_bytree': 0.8869339873610533,
                    'reg_alpha': 0.001,
                    'reg_lambda': 0.05,
                    'cat_smooth': 28.83549369109806,
                    'cat_l2': 7.399983164441014,
                    'min_data_per_group': 60,
                    'max_cat_threshold': 8,
                    'max_cat_to_onehot': 10,
                    'max_bin': 127,
                },
            ],
        },

        'search_spaces': {

        'xgboost': {
            'learning_rate': {
                'low': 0.05,
                'high': 0.12,
            },

            'max_depth': {
                'low': 2,
                'high': 4,
            },

            'subsample': {
                'low': 0.8,
                'high': 1.0,
            },

            'colsample_bytree': {
                'low': 0.8,
                'high': 1.0,
            },

            'reg_alpha': {
                'low': 0.0,
                'high': 0.2,
            },

            'reg_lambda': {
                'low': 0.5,
                'high': 3.0,
            },
        },

        'lightgbm': {
            'learning_rate': {
                'low': 0.03,
                'high': 0.15,
            },

            'max_depth': {
                'low': 3,
                'high': 7,
            },

            'num_leaves': {
                'low': 4,
                'high': 31,
            },

            'min_child_samples': {
                'low': 5,
                'high': 50,
                'step': 5,
            },

            'min_child_weight': {
                'low': 0.001,
                'high': 2.0,
            },

            'min_split_gain': {
                'low': 0.0,
                'high': 0.15,
            },

            'subsample_freq': [
                0,
                1,
            ],

            'subsample': {
                'low': 0.7,
                'high': 1.0,
            },

            'colsample_bytree': {
                'low': 0.7,
                'high': 1.0,
            },

            'reg_alpha': [
                0.0,
                0.001,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
                2.0,
            ],

            'reg_lambda': [
                0.0,
                0.001,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
                2.0,
            ],

            'cat_smooth': {
                'low': 5.0,
                'high': 30.0,
            },

            'cat_l2': {
                'low': 1.0,
                'high': 30.0,
            },

            'min_data_per_group': {
                'low': 20,
                'high': 120,
                'step': 10,
            },

            'max_cat_to_onehot': [
                4,
                10,
            ],

            'max_cat_threshold': [
                8,
                16,
                32,
            ],

            'max_bin': [
                31,
                63,
                127,
                255,
            ],
        },

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