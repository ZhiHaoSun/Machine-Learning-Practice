#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    N = 1000
    x = np.random.rand(N) * 8 - 4     # [-4,4)
    x.sort()
    corr1 = np.sin(x)
    corr2 = np.cos(0.3*x)
    y1 = corr1 + np.random.randn(N) * 0.05 # noise 
    y2 = corr2 + np.random.randn(N) * 0.05
    y = np.vstack((y1, y2)).T # 垂直合并，变成二维y
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 4
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(-4, 4, num=100).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.plot(corr1, corr2, 'r-', linewidth = 2, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=40, label='Depth=%d' % deep, alpha=0.6)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
