"""Classe abstraite : contrat commun à tous les modèles (polymorphisme)."""
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle."""

    @abstractmethod
    def predict(self, X):
        """Retourne des probabilités d'infection entre 0 et 1."""