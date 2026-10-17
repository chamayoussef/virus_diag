"""Point d'entrée : entraîne, évalue, puis diagnostique avec ClinicalPredictor."""
import sys
from pathlib import Path

from core import Dataset, LogisticRegression, NeuralNetwork
from pipeline import Trainer, Evaluator
from app import ClinicalPredictor
from app.api import FEATURES, diagnose_endpoint


def build(model, X_train, y_train, X_test, y_test, name):
    trainer = Trainer(model)                       # 1) entraînement
    trainer.train(X_train, y_train)
    scores = Evaluator(model, trainer.scaler).evaluate(X_test, y_test)  # 2) évaluation
    print(f"\n[{name}]")
    for k, v in scores.items():
        print(f"  {k}: {v if isinstance(v, tuple) else round(v, 3)}")
    return ClinicalPredictor(model, trainer.scaler)  # 3) prédiction (séparée)


if __name__ == "__main__":
    DATA = Path(__file__).parent / "data" / "patient_data.csv"
    ds = Dataset(DATA)
    X_train, X_test, y_train, y_test = ds.split()

    lr_pred = build(LogisticRegression(), X_train, y_train, X_test, y_test, "Régression logistique")
    nn_pred = build(NeuralNetwork(), X_train, y_train, X_test, y_test, "Réseau de neurones")

    # Même ClinicalPredictor, deux modèles différents (polymorphisme)
    sick = dict(zip(FEATURES, [60, 39.0, 1, 1, 1, 1, 91.0, 4.8, 55.0]))
    healthy = dict(zip(FEATURES, [30, 36.8, 0, 0, 0, 0, 98.0, 7.0, 4.0]))
    for name, p in [("LogReg", lr_pred), ("NN", nn_pred)]:
        print(f"\n{name} -> patient fiévreux : {diagnose_endpoint(p, sick)['diagnosis']}"
              f" | patient sain : {diagnose_endpoint(p, healthy)['diagnosis']}")

    if "--cli" in sys.argv:
        from app.interface_clinique import run
        run(nn_pred)