"""Interface clinique en ligne de commande."""
from .api import FEATURES, diagnose_endpoint

LABELS = {
    "age": "Âge (ans)",
    "temperature": "Température (°C)",
    "cough": "Toux (0 = non, 1 = oui)",
    "fatigue": "Fatigue (0 = non, 1 = oui)",
    "shortness_of_breath": "Essoufflement (0 = non, 1 = oui)",
    "loss_of_smell": "Perte d'odorat (0 = non, 1 = oui)",
    "spo2": "SpO2 - saturation en oxygène (%)",
    "wbc": "Globules blancs (10^9/L)",
    "crp": "CRP (mg/L)",
}


def run(predictor):
    print("=== Interface clinique - diagnostic viral ===")
    payload = {f: float(input(f"{LABELS[f]} : ")) for f in FEATURES}
    result = diagnose_endpoint(predictor, payload)
    print("Résultat :", result["diagnosis"])