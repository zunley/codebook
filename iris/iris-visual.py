#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


species_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
int_to_species = {v:k for k, v in species_to_int.items()}
colors = ["blue", "yellow", "green"]

def shuffle_data(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def split_data(X, y,test_size=0.4):
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


def preprocess(filepath: str):

    df = pd.read_csv(filepath)

    X_df = df.iloc[:, 1:-1] # 去掉第一列序号和最后一列标签
    y_df = df.iloc[:, -1] #  最后一列

    X = X_df.values
    y_label = y_df.values

    y = np.array([species_to_int[label] for label in y_label])

    # 打乱数据
    X, y = shuffle_data(X, y)

    # 分割训练测试集
    return split_data(X, y)


def generate_decision_boundary(X, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return xx, yy, Z
    
def main():
    X_train, y_train, X_test, y_test = preprocess('iris_reduced.csv')

    model = KNNClassifier(k=5)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy=calc_accuracy(y_pred, y_test)

    print("accuracy: %f" % (accuracy))

    # 可视化
    fig, ax = plt.subplots()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    # 绘制决策边界
    xx, yy, Z = generate_decision_boundary(X_test, model)
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)

    # 绘制数据点
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=cmap_bold, edgecolor='k', s=40)
    plt.show()



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
        y_pred = np.zeros(N, dtype=int)

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
