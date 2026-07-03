import random
import numpy as np


def set_seed(seed: int) -> None:
    '''Fix random seed for reproducible results.'''
    
    random.seed(seed)
    np.random.seed(seed)


import sys
from pathlib import Path


class TeeLogger:
    '''Write terminal output both to console and log file.'''

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_run_logging(config):
    '''Redirect terminal output to console and run log file.'''

    logs_dir = Path(config.paths.path_to_logs)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f'{config.general.experiment_name}.txt'

    tee_logger = TeeLogger(log_path)

    sys.stdout = tee_logger
    sys.stderr = tee_logger

    return tee_logger