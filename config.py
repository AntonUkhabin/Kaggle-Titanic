from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '48_l1_ratio_saga = 0.5, С=0.7',
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
            'C': 0.7,
            'l1_ratio': 0.5,
            'solver': 'saga',
            'max_iter': 500,
            'class_weight': None,
            'random_state': 0xC0FFEE,
        },
    },
}

config = OmegaConf.create(config)