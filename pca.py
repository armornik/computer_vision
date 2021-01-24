from pylab import *


def pca(X):
    """Метод главных компонент
    вход: матрица Х, в которой обучающие данные хранятся в виде линеаризованных массивов, по одному в каждой строке
    (например при помощи flatten)
    выход: матрица проекции (наиболее важные измерения в начале), дисперсия и среднее"""

    # получить количество измерений
    num_data, dim = X.shape

    # Центрировать данные
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA с компактным трюком
        M = dot(X, X.T)  # Ковариационная матрица
        e, EV = linalg.eigh(M)  # Собственные значения и собственные векторы
        tmp = dot(X.T, EV).T  # Компактный трюк
        V = tmp[::-1]  # Меняем порядок, потому что нам нужны последние собственные векторы
        S = sqrt(e)[::-1]  # Меняем порядок, потому что собственные значения перечислены в порядке возрастания
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA с использованием сингулярного разложения
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # Имеет смысл возвращать только первые num_data строк

    # Вернуть матрицу проекции, дисперсию и среднее
    return V, S, mean_X


def center(X):
    """    Center the square matrix X (subtract col and row means). """

    n, m = X.shape
    if n != m:
        raise Exception('Matrix is not square.')

    colsum = X.sum(axis=0) / n
    rowsum = X.sum(axis=1) / n
    totalsum = X.sum() / (n ** 2)

    # center
    Y = array([[X[i, j] - rowsum[i] - colsum[j] + totalsum for i in range(n)] for j in range(n)])

    return Y
