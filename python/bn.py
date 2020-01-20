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
        self.num_observables = x_train.shape[1]
        self.num_unknowns = y_train.shape[1]

        self._compute_apriori(dataset)
        self._compute_conditionals(dataset)

    def _compute_conditionals(self, dataset):
        self.conditionals = np.zeros(
            (2*self.num_unknowns, self.num_observables))

        for sample in dataset:
            for d_index, d_value in enumerate(sample[self.num_unknowns:]):
                if d_value == 1:
                    for g_index, g_value in enumerate(sample[:self.num_unknowns]):
                        self.conditionals[2*g_index + g_value, d_index] += 1

        for i in range(self.num_unknowns):
            self.conditionals[2*i] = self.conditionals[2*i] / \
                ((1-self.apriori[i]) * self.num_samples)
            self.conditionals[2*i+1] = self.conditionals[2 *
                                                         i+1] / (self.apriori[i] * self.num_samples)

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
        res = []
        for x in x_test:
            g_index = 0
            res.append(self._compute_prob(x, g_index))
        return np.array(res)

    def _compute_prob(self, x, g_index):
        prior = self.apriori[g_index]
        print('prior = ,', prior)

        likelihood = 1.0
        for event_index, event_value in enumerate(x):
            if event_index == 1:
                prob = self.conditionals[g_index+1, event_index]
            else:
                prob = (1 - self.conditionals[g_index+1, event_index])
            if prob == 0.0:
                prob = 0.00001
            likelihood *= prob
        print('likelihood = ,', likelihood)

        evidence = 1.0
        for event_index, event_value in enumerate(x):
            prob = self.apriori[event_index + self.num_unknowns]
            if event_value == 0:
                prob = 1 - prob
            evidence *= prob

        print('evidence = ,', evidence)

        return prior * likelihood / evidence


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
