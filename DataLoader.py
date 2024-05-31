import os
import gzip
import pickle
import numpy as np
import random

class ClassDataLoader:
    def __init__(self):
        pass
    def Load(self):
        path = 'data/mnist/mnist_28.pkl.gz'
        with gzip.open(path, 'rb') as f:
            # pickle을 사용하여 데이터 로드
            data = pickle.load(f, encoding='latin1')
        train, valid, test = data
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        # ========== ========== ==========
        def D2N(data): return np.array(data)
        def RS(data): return np.expand_dims(data.reshape(-1, 28, 28), axis=1)
        # ========== ========== ==========
        return train_x, train_y, valid_x, valid_y, test_x, test_y
