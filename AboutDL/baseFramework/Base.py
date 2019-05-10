#-*- coding: UTF-8 -*-
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import time
import math
import random
import pickle
import shutil
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D

from NNUtil import *

#数据生成器（DNN）
class Generator:
    def __init__(self, x, y, name="Generator", weights=None, n_class=None, shuffle=True):
        self._cache = {}
        self._x, self._y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if weights is None:
            self._sample_weights = None
        else:
            self._sample_weights = np.asarray(weights, np.float32)
        if n_class is not None:
            self.n_class = n_class
        else:
            y_int = self._y.astype(np.int32)
            if np.allclose(self._y, y_int):
                assert y_int.min() == 0, "标签应该从 0 开始"
                self.n_class = y_int.max() + 1
            else:
                self.n_class = 1
        self._name = name
        self._do_shuffle = shuffle
        self._all_valid_data = self._generate_all_valid_data()
        self._valid_indices = np.arange(len(self._all_valid_data))
        self._random_indices = self._valid_indices.copy()
        np.random.shuffle(self._random_indices)
        self._batch_cursor = -1

    def __enter__(self):
        self._cache_current_status()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_cache()

    def __getitem__(self, item):
        return getattr(self, "_" + item)

    def __len__(self):
        return self.n_valid

    def __str__(self):
        return "{}_{}".format(self._name, self.shape)

    __repr__ = __str__

    @property
    def n_valid(self):
        return len(self._valid_indices)

    @property
    def n_dim(self):
        return self._x.shape[-1]

    @property
    def shape(self):
        return self.n_valid, self.n_dim

    def _generate_all_valid_data(self):
        return np.hstack([self._x, self._y.reshape([-1, 1])])

    def _cache_current_status(self):
        self._cache["_valid_indices"] = self._valid_indices
        self._cache["_random_indices"] = self._random_indices

    def _restore_cache(self):
        self._valid_indices = self._cache["_valid_indices"]
        self._random_indices = self._cache["_random_indices"]
        self._cache = {}

    def set_indices(self, indices):
        indices = np.asarray(indices, np.int)
        self._valid_indices = self._valid_indices[indices]
        self._random_indices = self._random_indices[indices]

    def set_range(self, start, end=None):
        if end is None:
            self._valid_indices = self._valid_indices[start:]
            self._random_indices = self._random_indices[start:]
        else:
            self._valid_indices = self._valid_indices[start:end]
            self._random_indices = self._random_indices[start:end]

    def get_indices(self, indices):
        return self._get_data(np.asarray(indices, np.int))

    def get_range(self, start, end=None):
        if end is None:
            return self._get_data(self._valid_indices[start:])
        return self._get_data(self._valid_indices[start:end])

    def _get_data(self, indices, return_weights=True):
        data = self._all_valid_data[indices]
        if not return_weights:
            return data
        weights = None if self._sample_weights is None else self._sample_weights[indices]
        return data, weights

    def gen_batch(self, n_batch, re_shuffle=True):
        n_batch = min(n_batch, self.n_valid)
        logger = logging.getLogger("DataReader")
        if n_batch == -1:
            n_batch = self.n_valid
        if self._batch_cursor < 0:
            self._batch_cursor = 0
        if self._do_shuffle:
            if self._batch_cursor == 0 and re_shuffle:
                logger.debug("Re-shuffling random indices")
                np.random.shuffle(self._random_indices)
            indices = self._random_indices
        else:
            indices = self._valid_indices
        logger.debug("Generating batch with size={}".format(n_batch))
        end = False
        next_cursor = self._batch_cursor + n_batch
        if next_cursor >= self.n_valid:
            next_cursor = self.n_valid
            end = True
        data, w = self._get_data(indices[self._batch_cursor:next_cursor])
        self._batch_cursor = -1 if end else next_cursor
        logger.debug("Done")
        return data, w

    def gen_random_subset(self, n):
        n = min(n, self.n_valid)
        logger = logging.getLogger("DataReader")
        logger.debug("Generating random subset with size={}".format(n))
        start = random.randint(0, self.n_valid - n)
        subset, weights = self._get_data(self._random_indices[start:start + n])
        logger.debug("Done")
        return subset, weights

    def get_all_data(self, return_weights=True):
        if self._all_valid_data is not None:
            if return_weights:
                return self._all_valid_data, self._sample_weights
            return self._all_valid_data
        return self._get_data(self._valid_indices, return_weights)

#数据生成器（RNN）
class Generator3d(Generator):
    @property
    def n_time_step(self):
        return self._x.shape[1]

    @property
    def shape(self):
        return self.n_valid, self.n_time_step, self.n_dim

    def _generate_all_valid_data(self):
        return np.array([(x, y) for x, y in zip(self._x, self._y)])

#数据生成器（CNN）
class Generator4d(Generator3d):
    @property
    def height(self):
        return self._x.shape[1]

    @property
    def width(self):
        return self._x.shape[2]

    @property
    def shape(self):
        return self.n_valid, self.height, self.width, self.n_dim
