from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import cvxopt


class MySVM:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.w0 = None

    def fit(self, X, y):
        """Fit the model."""
        n = len(X)
        K = X.dot(X.T)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        G = cvxopt.matrix(np.identity(n) * -1)
        h = cvxopt.matrix(np.zeros(n))
        A = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.alpha = np.ravel(solution['x'])
        sv = self.alpha > 1e-5
        self.w = X[sv].T.dot(self.alpha[sv] * y[sv])
        self.w0 = y[sv][0] - self.w.dot(X[sv][0])


def load_data():
    """Return a 2D data set to demonstrate the features of linear SVM."""
    X, y = make_blobs(n_samples=200, centers=2,
                       random_state=0, cluster_std=0.5)
    y = y.astype(float)
    y[y == 0] = -1
    return X, y


def f(x, w, w0, c=0):
    """Return the x2 given x1, the model and the margin."""
    return (-w[0] * x - w0 + c) / w[1]


# load the data and fit the model
X, y = load_data()
svm = MySVM()
svm.fit(X, y)

# plot the data and the decision boundary
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y)
a0, b0 = min(X[:, 0]), max(X[:, 0])
a1 = f(a0, svm.w, svm.w0)
b1 = f(b0, svm.w, svm.w0)
plt.plot([a0, b0], [a1, b1], 'k')

# plot the two margins
a1 = f(a0, svm.w, svm.w0, -1)
b1 = f(b0, svm.w, svm.w0, -1)
plt.plot([a0, b0], [a1, b1], 'k--')

a1 = f(a0, svm.w, svm.w0, 1)
b1 = f(b0, svm.w, svm.w0, 1)
plt.plot([a0, b0], [a1, b1], 'k--')

plt.savefig("0.pdf")
