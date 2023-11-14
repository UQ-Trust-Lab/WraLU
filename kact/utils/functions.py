from numbers import Number
from typing import Union, Iterable

import numpy as np


def relu(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    return np.maximum(x, 0)


def drelu(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    return np.where(x > 0, 1, 0)


def sigmoid(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    ex = np.exp(-x)
    return ex / (1 + ex) ** 2


def tanh(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    return np.tanh(x)


def dtanh(x: Union[Number, np.ndarray, Iterable]) -> Union[Number, np.ndarray, Iterable]:
    tanhx = np.tanh(x)
    return 1 - tanhx ** 2
