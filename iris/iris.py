#!/usr/bin/env python3

import numpy as np
import pandas as pd

def shuffle_data(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def split_data(X, y,test_size=0.9):
    """
    使用 numpy 分割数据为训练集和测试集
    """

    test_size = int(len(X) * test_size)

    X_train = X[test_size:]
    y_train = y[test_size:]
    X_test = X[:test_size]
    y_test = y[:test_size]

    return X_train, y_train, X_test, y_test

def calc_accuracy(y_pred, y):
    return np.mean(y == y_pred)


def preprocess():

    df = pd.read_csv('iris_reduced.csv')

    X_df = df.iloc[:, :-1] # 去掉最后一列
    y_df = df.iloc[:, -1] #  最后一列

    X = X_df.values
    y_label = y_df.values

    label_to_int = {label: idx for idx, label in enumerate(sorted(set(y_label)))}
    y = np.array([label_to_int[label] for label in y_label])

    # 打乱数据
    X, y = shuffle_data(X, y)

    # 分割训练测试集

    return split_data(X, y)


def main():
    X_train, y_train, X_test, y_test = preprocess()

    model = KNNClassifier(k=5)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy=calc_accuracy(y_pred, y_test)

    print("accuracy: %f" % (accuracy))


def l2_distance(x, y):
    # sum((m, p), axis=1) ->  (m, )
    x_square_sum = np.sum(np.square(x), axis=1)
    # sum((n, p), axis=1) ->  (n, )
    y_square_sum = np.sum(np.square(y), axis=1)
    # (m, 1) + (1, n) -> (m, n) + (m, n) = (m, n)
    x_y_square_sum = x_square_sum[:, np.newaxis] + y_square_sum[np.newaxis, :]

    # (m, p) dot (p, n) = (m, n)
    x_y_dot = np.dot(x, y.T)

    # (m, n) -2(m, n) = (m, n)
    d = np.square(x_y_square_sum - 2*x_y_dot)

    return d


class KNNClassifier():
    def __init__(self, k=3):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        N = X_test.shape[0]
        y_pred = np.zeros(N)

        dists = l2_distance(X_test, self.X_train)

        for i in range(X_test.shape[0]):
            index = np.argsort(dists[i])
            index = index[:self.k]
            closest_y = self.y_train[index]

            values, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred


if __name__ == '__main__':
    main()
