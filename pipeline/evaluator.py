"""Evaluator : mesure la performance d'un modèle déjà entraîné."""
from utils import metrics


class Evaluator:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def evaluate(self, X_test, y_test):
        proba = self.model.predict(self.scaler.transform(X_test))
        y_pred = (proba >= 0.5).astype(int)
        tp, tn, fp, fn = metrics.confusion(y_test, y_pred)
        return {
            "accuracy": metrics.accuracy(y_test, y_pred),
            "precision": metrics.precision(y_test, y_pred),
            "recall": metrics.recall(y_test, y_pred),
            "f1": metrics.f1_score(y_test, y_pred),
            "confusion (TP,TN,FP,FN)": (tp, tn, fp, fn),
        }