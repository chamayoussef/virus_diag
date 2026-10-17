"""Couche application : ClinicalPredictor + point d'accès API.

ClinicalPredictor reçoit un modèle DÉJÀ entraîné (injection de dépendance)
et ne dépend que de sa méthode predict() : polymorphisme / duck typing.
"""
import numpy as np

# Même ordre que les colonnes de patient_data.csv
FEATURES = ["age", "temperature", "cough", "fatigue",
            "shortness_of_breath", "loss_of_smell", "spo2", "wbc", "crp"]


class ClinicalPredictor:
    def __init__(self, model, scaler=None, threshold=0.5):
        self.model = model          # n'importe quel objet avec predict()
        self.scaler = scaler        # même prétraitement qu'à l'entraînement
        self.threshold = threshold

    def diagnose(self, patient_data):
        x = np.asarray(patient_data, dtype=float).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        prediction = float(np.ravel(self.model.predict(x))[0])
        return "Infecté" if prediction >= self.threshold else "Sain"


def diagnose_endpoint(predictor, payload):
    """Reçoit un dict de données patient, renvoie un dict de réponse."""
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return {"status": "error", "missing_fields": missing}
    values = [payload[f] for f in FEATURES]
    return {"status": "ok", "diagnosis": predictor.diagnose(values)}