"""Chargement et découpage des données patients."""
import csv
import numpy as np


class Dataset:
    def __init__(self, path, target="infected"):
        self.path = path
        self.target = target
        self.feature_names = []
        self.X = None
        self.y = None
        self._load()

    def _load(self):
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            self.feature_names = [c for c in reader.fieldnames if c != self.target]
            X, y = [], []
            for row in reader:
                X.append([float(row[c]) for c in self.feature_names])
                y.append(float(row[self.target]))
        self.X = np.array(X)
        self.y = np.array(y)

    def split(self, test_size=0.2, seed=42):
        """Mélange les patients puis sépare train / test."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.X))
        cut = int(len(idx) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
        return self.X[tr], self.X[te], self.y[tr], self.y[te]

    def __len__(self):
        return len(self.X)