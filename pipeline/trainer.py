"""Trainer : responsable UNIQUEMENT de l'entraînement."""
import pickle
from utils.preprocessing import Standardizer


class Trainer:
    def __init__(self, model):
        self.model = model
        self.scaler = Standardizer()

    def train(self, X_train, y_train):
        Xs = self.scaler.fit_transform(X_train)   # le scaler apprend sur le train
        self.model.fit(Xs, y_train)
        return self.model

    def save(self, path):
        """Sauvegarde le modèle entraîné et son scaler."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)