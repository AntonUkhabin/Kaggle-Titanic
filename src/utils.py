import random
import numpy as np


def set_seed(seed: int) -> None:
    '''Fix random seed for reproducible results.'''
    
    random.seed(seed)
    np.random.seed(seed)