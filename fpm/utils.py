"""This module contains some useful tools."""

from numbers import Number
import numpy as np


def isnumeric(a):
    return isinstance(a, Number) and (not np.isinf(a))


def isint(a):
    return isinstance(a, int)
