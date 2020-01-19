import pomegranate
from load_data import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def get_model(data):
    structure = ((), (), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    model = pomegranate.BayesianNetwork.from_structure(data, structure)
    return model


def main():
    x_data, y_data = get_data()

    # cv = KFold(n_splits=10, shuffle=True)
    # folds_acc = []

    # for train_index, test_index in cv.split(x_data, y_data):
    #     x_train, x_test = x_data[train_index], x_data[test_index]
    #     y_train, y_test = y_data[train_index], y_data[test_index]

    #     all_variables = np.hstack((y_train, x_train))
    #     model = get_model(all_variables)

    #     x_test_all_variables = np.hstack(
    #         (np.ones_like(y_test) * np.nan, x_test))
    #     y_pred = model.predict(x_test_all_variables)
    #     y_pred = np.vstack(y_pred)
    #     y_pred = y_pred.astype('float64')
    #     y_pred = y_pred[:, :2]

    #     acc = accuracy_score(y_test, y_pred)
    #     if not np.isclose(acc, 1.0):
    #         print('acc not 1.0, but eq ', acc)
    #         print(np.sort(y_train, axis=0))
    #     folds_acc.append(acc)

    # print('folds_acc = ', np.mean(folds_acc))

    all_variables = np.hstack((y_data, x_data))
    network = get_model(all_variables)

    x_test = np.hstack((np.ones((2, 2)) * np.nan, x_data[:2, :]))
    print('x_test = \n', x_test)
    print('ground thruth for x_test = \n', y_data[:2, :])
    y_hat = network.predict_proba(x_test)
    print('y_hat = ', y_hat)


if __name__ == '__main__':
    main()
