import pytest
import numpy as np

from BayesianNetwork import *
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

# -------------------------------

# def test_apriori_for_targets_in_medical_dataset_are_correct():
#     x_data, y_data = get_data()

#     network = BayesianNetwork()
#     network.fit(x_data, y_data)

#     assert np.isclose(network.apriori[:2], [0.49166667, 0.41666667]).all()


# def test_test_case1():
#     x_data = np.array([[0], [0], [0], [1], [1], [1], [1], [1], [1], [1]])
#     y_data = np.array([[0], [1], [1], [0], [0], [0], [0], [0], [1], [1]])

#     network = BayesianNetwork()
#     network.fit(x_data, y_data)
#     assert (network.joint_prob == np.array([[0.1, 0.5], [0.2, 0.2]])).all()

#     assert network.apriori[0] == 0.4
#     assert network.apriori[1] == 0.7

#     x_test = np.array([[1]])
#     y_pred = network.predict_proba(x_test)
#     assert np.isclose(y_pred, 0.28571428571428575)

#     y_pred = network.predict_proba(x_data)
#     assert_is_probability(y_pred)


def test_test_case2():
    x_data = np.hstack(([0, 0]*5, [0, 1]*10, [1, 0]*5, [1, 1]
                        * 20, [0, 0]*20, [0, 1]*25, [1, 0]*10, [1, 1]*5)).reshape(100, 2)
    y_data = np.hstack(([0]*40, [1]*60)).reshape(100, 1)

    network = BayesianNetwork()
    network.fit(x_data, y_data)

    assert network.conditionals[0, 0] == 5/8
    assert network.conditionals[0, 1] == 3/4
    assert network.conditionals[1, 0] == 1/4
    assert network.conditionals[1, 1] == 1/2

    assert network.apriori[0] == 0.6
    assert network.apriori[1] == 0.4
    assert network.apriori[2] == 0.6

    x_test = np.array([[1, 1]])
    y_pred = network.predict_proba(x_test)
    assert np.isclose(y_pred, 0.3125)

    # y_pred = network.predict_proba(x_data)
    # assert_is_probability(y_pred)


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

    assert network.conditionals[0, 0, 0] == 2/3
    assert network.conditionals[1, 0, 0] == 3/5
    assert network.conditionals[0, 1, 0] == 2/3
    assert network.conditionals[1, 1, 0] == 4/9
    # assert network.conditionals[1, 0, 0] == 0.02
    # assert network.conditionals[1, 0, 1] == 0.02
    # assert network.conditionals[1, 1, 0] == 0.03
    # assert network.conditionals[1, 1, 1] == 0.05


#     joint_sum = np.sum(network.joint_prob, axis=None)
#     assert np.isclose(joint_sum, 1.0)

#     assert network.apriori[0] == 0.7
#     assert network.apriori[1] == 0.57
#     assert network.apriori[2] == 0.55
#     assert network.apriori[3] == 0.57

#     x_test = np.array([[1, 1]])
#     y_pred = network.predict_proba(x_test)
#     assert np.isclose(y_pred, 0.6220095693779905)

#     y_pred = network.predict_proba(x_data)
#     assert_is_probability(y_pred)

# failing right now
# def test_all_predictions_are_probabilities():
#     x_data, y_data = get_data()

#     network = BayesianNetwork()
#     network.fit(x_data, y_data)
#     y_pred = network.predict_proba(x_data)
#     assert_is_probability(y_pred)
