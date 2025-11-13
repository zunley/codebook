#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_iris(filepath: str):
    """
    加载 iris 数据集，并返回数据和标签。

    Args:
        filepath (str): 数据集路径

    Returns:
        X (np.ndarray): 数据矩阵
        y (np.ndarray): 标签矩阵
    """
    df = pd.read_csv(filepath)
    X_df = df.iloc[:, :-1]  # 去掉最后一列
    y_df = df.iloc[:, -1]   #  最后一列

    X = X_df.values
    y_label = y_df.values

    return X, y_label

def perform_pca(X, k=2):
    """
    对输入数据进行主成分分析（PCA）。

    Args:
        X (np.ndarray): 原始特征数据
        k (int): 保留的主成分数量

    Returns:
        X_reduced (np.ndarray): 降维后的数据，形状为 [n_samples, k]。
    """
    # X shape: [n_samples, n_features]
    N = X.shape[0]
    X_centered = X - np.mean(X, axis=0)

    cov = np.dot(X_centered.T, X_centered) / (N - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # W shape: [n_features, k]
    W = eigenvectors[:, :k]
    X_reduced = np.dot(X_centered, W)

    return X_reduced

def group_by_species(X, y):
    """
    根据标签对样本进行分组。

    Args:
        X : np.ndarray
            样本数据
        y : np.ndarray
            标签数据

    Returns:
        dict[str: np.ndarray]
            分组数据
    """
    return {label: X[y == label] for label in np.unique(y)}


def plot_species_scatter(groups: dict):
    for label in groups:
        xy = groups[label]
        plt.scatter(xy[:, 0], xy[:, 1], label=label)
    plt.show()


def save_csv(data: np.ndarray, filepath: str):
    df = pd.DataFrame(data)
    df.to_csv(filepath)

def main():
    X, y = load_iris('iris.csv')
    # X_reduced shape: [n_samples, 2]
    X_reduced = perform_pca(X, 2)

    iris_reduced = np.concatenate((X_reduced, y.reshape(150, 1)), axis=1)
    #save_csv(iris_reduced, 'iris_reduced.csv')
    xmax, ymax = X_reduced[:, 0].max() + 1, X_reduced[:, 1].max() + 1
    xmin, ymin = X_reduced[:, 0].min() - 1, X_reduced[:, 1].min() - 1
    
    xx = np.arange(xmin, xmax, 0.1)
    yy = np.arange(ymin, ymax, 0.01)
    print(xx.shape)
    print(yy.shape)
    
    species_groups = group_by_species(X_reduced, y)
    plot_species_scatter(species_groups)

if __name__ == "__main__":
    main()
