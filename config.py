from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': 'l1_C_0.7_liblinear',
    },
    'paths': {
        'path_to_csv':'./data/train.csv',
        'path_to_kaggle_test':'./data/test.csv',
        'path_to_submission':'./outputs/submission.csv',
        'path_to_experiments': './outputs/experiments.csv',
        'path_to_coefficients': './outputs/coefficients.csv',
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
        'name': 'logistic_regression',
        'params': {
            'C': 1.0,
            'l1_ratio': 0.0,
            'solver': 'lbfgs',
            'max_iter': 500,
            'class_weight': None,
        },
    },
}

config = OmegaConf.create(config)