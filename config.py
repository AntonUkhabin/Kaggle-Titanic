from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': 'Baseline',
    },
    'paths': {
        'path_to_csv':'./data/train.csv',
        'path_to_kaggle_test':'./data/test.csv',
        'path_to_submission':'./outputs/submission.csv',
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
}

config = OmegaConf.create(config)