from scipy.optimize import minimize
import numpy as np, functools as func
from sklearn.tree import DecisionTreeRegressor


class BinaryBoostingClassifier:
    def __init__(self, n_estimators, lr):
        self.lr = lr
        self.n_est = n_estimators

    def fit(self, X, y):
        self.ens = []
        for i in range(self.n_est):
            # Eval the gradients dL(y, y_pred)/dy_pred for L2-loss
            # L2-loss is not the best choice for classification but the simplest one :)
            # Try CrossEntopy as an assignment!
            grad = y - self._predict(X)
            # Fit the next estimator to predict **gradients**.
            # This is the regression problem.
            estimator = DecisionTreeRegressor(max_depth=3).fit(X, grad)
            self.ens.append(estimator)  # Append the estimator to ensemble
        return self

    def _predict(self, X):
        # initial approximation of y_pred by zero predictions
        if len(self.ens) == 0:
            return np.zeros(len(X))
        else:
            return func.reduce(lambda a, b: a + b, [self.lr * e.predict(X) for e in self.ens])

    def predict(self, X):
        return np.array(self._predict(X) > 0.5).astype(int)


class AdaBoost(object):

    def __init__(self, esti_num=10):
        self.esti_num = esti_num
        self.estimators = []
        self.alphas = []

    def fit(self, x, y):
        n_data = x.shape[0]
        w = np.ones(x.shape[0]) / n_data
        eps = 1e-16
        prediction = np.zeros(n_data)
        for i in range(self.esti_num):
            self.estimators.append(DecisionTree(
                metric_type='Gini impurity', depth=2))
            self.estimators[i].fit(x, y, w)
            pred_i = self.estimators[i].predict(x)
            error_i = (pred_i != y).dot(w.T)
            self.alphas.append(np.log((1.0 - error_i) / (error_i + eps)) / 2)
            w = w * np.exp(self.alphas[i] * (2 * (pred_i != y) - 1))
            w = w / w.sum()

            prediction += pred_i * self.alphas[i]
            print("Tree {} constructed, acc {}".format(
                i, (np.sign(prediction) == y).sum() / n_data))

    def predict(self, x):
        return sum(esti.predict(x) * alpha for esti, alpha in zip(self.estimators, self.alphas))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):

    def __init__(self):
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.decay = 1 - 1e-4

    def loss(self, x, y):  # using cross entropy as loss function
        eps = 1e-20
        h = self.predict(x)
        return -(np.multiply(y, np.log(h + eps)) + np.multiply((1 - y), np.log(1 - h + eps))).mean()

    def fit(self, x, y):
        label_num = len(np.unique(y))
        labels = np.zeros((x.shape[0], label_num))
        labels[np.arange(x.shape[0]), y] = 1
        self.w = np.random.randn(x.shape[1], label_num)
        self.b = np.random.randn(1, label_num)
        self.mom_w = np.zeros_like(self.w)
        self.mom_b = np.zeros_like(self.b)

        train_num = x.shape[0]
        for i in range(5000):
            h = sigmoid(x.dot(self.w) + self.b)
            g_w = x.T.dot(h - labels) / train_num
            g_b = (h - labels).sum() / train_num
            self.mom_w = self.gamma * self.mom_w + self.learning_rate * g_w
            self.w = (self.w - self.mom_w) * self.decay
            self.mom_b = self.gamma * self.mom_b + self.learning_rate * g_b
            self.b = (self.b - self.mom_b) * self.decay
            if i % 100 == 0:
                print(self.loss(x, labels))

    def predict(self, x):
        return sigmoid(x.dot(self.w) + self.b)