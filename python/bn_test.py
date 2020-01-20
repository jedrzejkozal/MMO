import pytest
import numpy as np

from bn import *
from load_data import *


def assert_is_probability(matrix):
    assert (matrix >= 0.0).all() and (matrix <= 1.0).all()


def test_each_elem_of_conditionals_is_between_0_and_1():
    x_data = np.random.randint(0, 2, size=(100, 8))
    y_data = np.random.randint(0, 2, size=(100, 3))
    sut = BayesianNetwork()
    sut.fit(x_data, y_data)

    assert_is_probability(sut.conditionals)


def test_each_elem_of_apriori_is_between_0_and_1():
    x_data = np.random.randint(0, 2, size=(100, 8))
    y_data = np.random.randint(0, 2, size=(100, 3))
    sut = BayesianNetwork()
    sut.fit(x_data, y_data)

    assert_is_probability(sut.apriori)


def test_test_case3():
    x_data = np.hstack((
        [0, 0]*5, [0, 1]*1, [1, 0]*2, [1, 1]*10,
        [0, 0]*2, [0, 1]*2, [1, 0]*3, [1, 1]*5,
        [0, 0]*6, [0, 1]*4, [1, 0]*10, [1, 1]*5,
        [0, 0]*5, [0, 1]*20, [1, 0]*10, [1, 1]*10,
    )).reshape(100, 2)
    y_data = np.hstack((
        [0, 0]*18,
        [0, 1]*12,
        [1, 0]*25,
        [1, 1]*45,
    )).reshape(100, 2)

    network = BayesianNetwork()
    network.fit(x_data, y_data)

    print(network.conditionals)

    assert network.conditionals[0, 0] == 2/3
    assert network.conditionals[1, 0] == 1/2
    assert network.conditionals[1, 1] == 39/70
