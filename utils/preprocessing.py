"""Prétraitement : standardisation des variables."""
import numpy as np


class Standardizer:
    """Transforme chaque colonne pour avoir moyenne 0 et écart-type 1."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0   # évite la division par zéro
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)