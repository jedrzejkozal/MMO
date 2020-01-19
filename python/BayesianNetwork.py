import numpy as np

from load_data import *


class BayesianNetwork:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        """
        dataset is matrix of shape (num_samples, num_variables) with only zeros and ones
        (this class is suited case for binary variables only)
        joint_prob is vector where each element is joint probability according to schema:
        g0 = 0, g1=0, ..., gk=0, d0=0, ..., dk-1=0 dk=0
        g0 = 0, g1=0, ..., gk=0, d0=0, ..., dk-1=0 dk=1
        in other word occurences of events are encoded as binary number, wich is index of join_prob vector
        apriori vector have num_features elements, and each element correspond to apriori from marginal:
        apriori = [P(g0=1), P(g1=1), ..., P(gk=1), P(d0=1), ..., P(dk=1)]
        """
        dataset = np.hstack((y_train, x_train))
        dataset = dataset.astype('int')
        self.num_samples = dataset.shape[0]
        self.num_variables = dataset.shape[1]
        self._compute_joint(dataset)
        self._compute_apriori(dataset)

    def _compute_joint(self, dataset):
        self.joint_prob = np.zeros((2**self.num_variables,), dtype='float64')

        # print('\n')
        for sample in dataset:
            index = self._vector_to_int(sample)
            # print('sample = {}, index = {}'.format(sample, index))
            self.joint_prob[index] += 1.0

        # print('num_samples = ', self.num_samples)
        self.joint_prob = self.joint_prob / self.num_samples
        new_shape = tuple(2 for _ in range(self.num_variables))
        reshaped = self.joint_prob.reshape(new_shape)
        # print('self.joint_prob[0] = ', self.joint_prob[0])
        self._check_reshaped(reshaped)
        self.joint_prob = reshaped

    def _check_reshaped(self, reshaped):
        for index in range(len(self.joint_prob)):
            binary = '{0:0{1}b}'.format(index, self.num_variables)
            i = [int(bit, 2) for bit in binary]
            assert self.joint_prob[index] == reshaped[tuple(i)]

    def _vector_to_int(self, vector):
        vector_str = [str(elem) for elem in vector]
        string = ''.join(vector_str)
        return int(string, 2)

    def _compute_apriori(self, dataset):
        self.apriori = np.zeros((self.num_variables,))
        for variable_index in range(self.num_variables):
            self.apriori[variable_index] = self._compute_marginal(
                dataset, variable_index)

    def _compute_marginal(self, dataset, variable_index):
        num_positive_examples = len(np.argwhere(
            dataset[:, variable_index] == 1.0))
        return num_positive_examples / self.num_samples

    def predict(self, x_test):
        # P(g0=1) only for now
        probs = self.predict_proba(x_test)
        return np.array(probs > 0.5).astype('int')

    def predict_proba(self, x_test):
        self.num_unknowns = self.num_variables - x_test.shape[1]

        res = []
        for x in x_test:
            g_index = 0
            # nominator = self._compute_nominator(x, g_index)
            # denominator = self._compute_denominator(x, g_index)
            # print('nominator = ', nominator)
            # print('denominator = ', denominator)
            # print('nominator no log = ',
            #       self._compute_nominator_no_log(x, g_index))
            # print('denominator no log = ',
            #       self._compute_denominator_no_log(x, g_index))
            # res.append(nominator / denominator)
            res.append(self.compute(x, g_index))
        return np.array(res)

    def _compute_nominator(self, x, g_index):
        log_sum = 0.0
        for event_index, event_value in enumerate(x):
            indexes = [slice(None) for _ in range(self.num_variables)]
            indexes[g_index] = 1
            indexes[event_index + self.num_unknowns] = int(event_value)
            indexes = tuple(indexes)
            assert self.joint_prob[indexes].size == 2**(self.num_variables-2)
            joint_sum = np.sum(self.joint_prob[indexes], axis=None)
            # if joint_sum == 0:
            #     print('event_index with zero prob = ', event_index,
            #           event_index + self.num_unknowns)
            #     print('event value with zero prob = ', event_value)
            #     print(self.joint_prob[indexes])
            log_sum += np.log(joint_sum)
        nominator = log_sum - \
            (self.num_variables-self.num_unknowns-1) * \
            np.log(self.apriori[g_index])
        return np.exp(nominator)

    def _compute_denominator(self, x, g_index):
        log_sum = 0.0
        for event_index, event_value in enumerate(x):
            event_prob = self.apriori[event_index + self.num_unknowns]
            if event_value == 0:
                event_prob = 1 - event_prob
            log_sum += np.log(event_prob)
        return np.exp(log_sum)

    def _compute_nominator_no_log(self, x, g_index):
        sum = 1.0
        g_apriori = self.apriori[g_index]
        for event_index, event_value in enumerate(x):
            indexes = [slice(None) for _ in range(self.num_variables)]
            indexes[g_index] = 1
            indexes[event_index + self.num_unknowns] = int(event_value)
            indexes = tuple(indexes)
            # print(indexes)
            assert self.joint_prob[indexes].size == 2**(self.num_variables-2)
            conditional_prob = np.sum(
                self.joint_prob[indexes], axis=None) / g_apriori
            sum *= conditional_prob
        print('_compute_nominator_no_log = ', sum)
        return sum * g_apriori

    def _compute_denominator_no_log(self, x, g_index):
        sum = 1.0
        for event_index, event_value in enumerate(x):
            event_prob = self.apriori[event_index + self.num_unknowns]
            if event_value == 0:
                event_prob = 1 - event_prob
            sum *= event_prob
        print('_compute_denominator_no_log = ', sum)
        return sum

    def compute(self, x, g_index):
        prob = 1.0
        for event_index, event_value in enumerate(x):
            indexes = [slice(None) for _ in range(self.num_variables)]
            indexes[g_index] = 1
            indexes[event_index + self.num_unknowns] = int(event_value)
            indexes = tuple(indexes)
            joint_prob = np.sum(self.joint_prob[indexes], axis=None)

            event_prob = joint_prob / self.apriori[g_index]

            if event_prob >= 1.0:
                print('event_prob = ', event_prob)

            prob *= event_prob

        indexes = [slice(None) for _ in range(self.num_variables)]
        for event_index, event_value in enumerate(x):
            indexes[event_index + self.num_unknowns] = int(event_value)
        indexes = tuple(indexes)
        joint_prob = np.sum(self.joint_prob[indexes], axis=None)
        prob /= joint_prob

        return prob * self.apriori[g_index]


def main():
    x_data, y_data = get_data()

    network = BayesianNetwork()
    network.fit(x_data, y_data)
    print(network.apriori)

    x_test = x_data[:2, :]
    print('x_test = \n', x_test)
    print('ground thruth for x_test = \n', y_data[:2, :])
    y_hat = network.predict_proba(x_test)
    print('y_hat = ', y_hat)


if __name__ == '__main__':
    main()
