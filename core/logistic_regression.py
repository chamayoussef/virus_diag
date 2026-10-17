"""Régression logistique implémentée avec numpy."""
import numpy as np
from .model import Model
from .optimizer import GradientDescent


def sigmoid(z):
    """Transforme un score en probabilité entre 0 et 1."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(Model):
    def __init__(self, lr=0.1, epochs=500):
        self.optimizer = GradientDescent(lr)
        self.epochs = epochs
        self.w = None
        self.b = np.zeros(1)

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)      # probabilités prédites
            err = p - y                           # erreur
            grad_w = X.T @ err / n                # gradient des poids
            grad_b = np.array([err.mean()])       # gradient du biais
            self.optimizer.step([self.w, self.b], [grad_w, grad_b])
        return self

    def predict(self, X):
        return sigmoid(X @ self.w + self.b)