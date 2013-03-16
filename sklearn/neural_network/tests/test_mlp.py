from sklearn.neural_network import MLPClassifier

from sklearn import datasets

import numpy as np


def test_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    #X = X[y != 2]
    #y = y[y != 2]

    clf = MLPClassifier(n_hidden=100, learning_rate=.07, max_iter=1000,
                        alpha=.01, tol=1e-4, verbose=True, random_state=37)
    clf.fit(X, y)
    print(np.mean(clf.predict(X) == y))
