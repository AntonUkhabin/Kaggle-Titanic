from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '82_knn_remove_embarked_isalone',
    },
    'paths': {
        'path_to_csv':'./data/train.csv',
        'path_to_kaggle_test':'./data/test.csv',
        'path_to_submission':'./outputs/submission.csv',
        'path_to_experiments': './outputs/experiments.csv',
        'path_to_coefficients': './outputs/coefficients.csv',
        'path_to_logs': './outputs/logs',
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
        'active': 'knn',

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
        },
    },
}

config = OmegaConf.create(config)