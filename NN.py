#@Peimou Sun

import numpy as np
import numba as nb


@nb.njit()
def l1_norm(x, y):
    return np.sum(np.abs(x - y), axis = 1)

@nb.njit()
def l2_norm(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2), axis = 1))

@nb.njit()
def linfty_norm(x, y):
    res = np.empty((len(y)))
    for i in range(len(y)):
        res[i] = np.max(np.abs(x - y[i]))
    return res


@nb.njit()
def dist_class(x, dataX, datay, K, method):
    '''
    Life is short, code in Numba.

    Parameters:
    -----------
    x: 2D np.ndarray. The testing samples
    dataX: 2D np.ndarray. The training samples.
    datay: 1D np.ndarray. The label of training samples.
    method: distance function. If you want to add new distance function, write it in numba.

    Returns:
    --------
    class result

    '''
    N = len(x)
    res = np.empty(N)
    for i in range(N):
        dis = method(x[i], dataX)
        ind = np.argsort(dis)[:K]
        label = datay[ind]
        res[i] = np.argmax(np.bincount(label))
    return res


class NN(object):
    def __init__(self, N):
        self.N = N

    @staticmethod
    def l2_norm(x, y):
        return np.sqrt(np.sum(np.power(x-y, 2), axis=1))

    @staticmethod
    def l1_norm(x, y):
        return np.sum(np.abs(x - y), axis=1)

    @staticmethod
    def linfty_norm(x, y):
        return np.max(np.abs(x - y), axis=1)


    def fit(self, X, y):
        try:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
        except:
            X = np.array([[X]])

        self.X = X
        self.y = y


    def predict(self, X, method):
        try:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
        except:
            X = np.array([[X]])
        return dist_class(X, self.X, self.y, self.N, method= method)


if __name__ == "__main__":

    x = np.array([1,2,3])
    y = np.array([[4,5,6], [1,2,3]])
    nn = NN(1)
    print(l1_norm(x, y))
    print(l2_norm(x, y))
    print(linfty_norm(x, y))


    a = np.random.normal(5, 1, size=100000)
    la = [1] * 100000
    b = np.random.normal(0, 1, size=100000)
    lb = [0] * 100000
    data = np.hstack([a, b]).reshape(-1,1)
    label = np.hstack([la, lb])
    test = np.linspace(-1,6,2000).reshape(-1,1)
    nn = NN(1)
    nn.fit(data, label)
    res = nn.predict(test, l1_norm)