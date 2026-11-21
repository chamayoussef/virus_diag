"""K plus proches voisins (KNN) : un nouveau modèle pour virus_diag."""
import numpy as np
from .model import Model


class KNN(Model):
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        # KNN ne "calcule" rien : il mémorise les patients connus
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # distances euclidiennes de chaque patient à tous les patients connus
        dist = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
        nearest = np.argsort(dist, axis=1)[:, :self.k]
        # probabilité d'infection = part de voisins infectés
        return self.y[nearest].mean(axis=1)