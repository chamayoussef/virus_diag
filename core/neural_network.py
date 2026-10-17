"""Petit réseau de neurones (1 couche cachée) avec numpy."""
import numpy as np
from .model import Model
from .optimizer import GradientDescent
from .logistic_regression import sigmoid


class NeuralNetwork(Model):
    def __init__(self, hidden=8, lr=0.1, epochs=800, seed=0):
        self.hidden = hidden
        self.optimizer = GradientDescent(lr)
        self.epochs = epochs
        self.seed = seed

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.5, (d, self.hidden))   # entrée -> couche cachée
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.5, self.hidden)        # couche cachée -> sortie
        self.b2 = np.zeros(1)

    def _forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)               # couche cachée
        p = sigmoid(h @ self.W2 + self.b2)               # probabilité de sortie
        return h, p

    def fit(self, X, y):
        n, d = X.shape
        self._init_params(d)
        for _ in range(self.epochs):
            h, p = self._forward(X)
            # rétropropagation du gradient
            d2 = (p - y) / n
            gW2 = h.T @ d2
            gb2 = np.array([d2.sum()])
            dh = np.outer(d2, self.W2) * (1 - h ** 2)
            gW1 = X.T @ dh
            gb1 = dh.sum(axis=0)
            self.optimizer.step([self.W1, self.b1, self.W2, self.b2],
                                [gW1, gb1, gW2, gb2])
        return self

    def predict(self, X):
        return self._forward(X)[1]