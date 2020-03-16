import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal
import math


class MultiGuassian(object):
    def __inti__(self):
        self.mu = np.nan
        self.sigma = np.nan
        self.py = {}

    @staticmethod
    def pdf(data, mu, sigma):
        try:
            data = np.atleast_2d(data)
            data = data.reshape(-1,1) - mu
            sigma_det = np.linalg.det(sigma)
            N = len(data)
            sigma_inv = np.linalg.inv(sigma)
            sc = data.T@sigma_inv@data
            return 1/(np.sqrt(np.power(2*math.pi, N) * sigma_det)) * np.exp(-sc/2)
        except:
            mu = mu.T[0]
            return multivariate_normal.pdf(data[0], mean = mu,
                                           cov = sigma, allow_singular=True)


    def fit(cls, data:np.ndarray, label:np.ndarray):
        '''
        Parameters:
        -----------
        data: 1D or 2D np.ndarray
        label: 1D np.ndarray

        Returns:
        --------
        The mean, var and probability of each class

        '''
        if len(data.shape)>2: raise ValueError(f"Need 2D array, get{len(data.shape)}")
        try:
            N, ndim = data.shape
        except ValueError:
            N, ndim = data.shape[0], 1
        cla = Counter(label)
        mu = {con:np.mean(data[label == con], axis = 0).reshape(-1,1) for con in cla.keys()}
        sigma = sum([(data[i].reshape(-1,1) - mu[label[i]])@
                     (data[i].reshape(-1,1) - mu[label[i]]).T for i in range(N)])/N
        cls.mu = mu
        cls.sigma = np.array(sigma)
        cls.py = {con[0]: con[1]/N for con in cla.items()}
        return cls.mu, cls.sigma, cls.py


    def predict(cls, data:np.ndarray):
        '''
        Parameters:
        -----------
        data: this function only support one sample

        Returns:
        --------
        class based on Gaussian Discriminator

        '''
        key = list(cls.mu.keys())
        p = [cls.pdf(data, cls.mu[con], cls.sigma) * cls.py[con] for con in key]
        ind = np.argmax(p)
        return key[ind]



#test
if __name__ == "__main__":
    a = np.random.normal(5, 1, size = 1000)
    la = [1]*1000
    b = np.random.normal(0, 1, size = 1000)
    lb = [0]*1000
    data = np.hstack([a, b])
    label = np.hstack([la, lb])
    mg = MultiGuassian()
    mg.fit(data,label)
    print("mu = ", mg.mu)
    print("sigma = ", mg.sigma)
    print("py = ", mg.py)
    print(mg.predict(-0.22))
