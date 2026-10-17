# virus_diag — Diagnostic viral par IA

Projet pédagogique d'architecture IA orientée objet : un modèle (régression logistique ou réseau de neurones, écrits avec numpy) prédit si un patient est **Infecté** ou **Sain** à partir de 9 variables cliniques.

> **Avertissement :** les données sont **synthétiques** (générées par `generate_data.py`). Ce projet est un exercice et ne doit pas être utilisé pour un vrai diagnostic médical.

## Arborescence

```
virus_diag/
├── data/
│   └── patient_data.csv          # 600 patients générés
├── core/
│   ├── dataset.py                # chargement + découpage train/test
│   ├── model.py                  # classe abstraite Model (fit, predict)
│   ├── logistic_regression.py    # régression logistique
│   ├── neural_network.py         # réseau de neurones (1 couche cachée)
│   └── optimizer.py              # descente de gradient
├── pipeline/
│   ├── trainer.py                # entraînement
│   └── evaluator.py              # évaluation
├── utils/
│   ├── preprocessing.py          # standardisation
│   └── metrics.py                # accuracy, precision, recall, F1
├── app/
│   ├── api.py                    # ClinicalPredictor + point d'accès API
│   └── interface_clinique.py     # interface en ligne de commande
├── generate_data.py              # génération des données
├── main.py                       # point d'entrée
└── requirements.txt
```

## Installation et utilisation

```powershell
pip install -r requirements.txt
python generate_data.py      # (re)génère data/patient_data.csv
python main.py               # entraîne, évalue et teste les deux modèles
python main.py --cli         # interface clinique interactive
```

## Variables du patient

| Variable | Signification |
|---|---|
| `age` | Âge en années |
| `temperature` | Température corporelle (°C) |
| `cough` | Toux (0 = non, 1 = oui) |
| `fatigue` | Fatigue (0 = non, 1 = oui) |
| `shortness_of_breath` | Essoufflement (0 = non, 1 = oui) |
| `loss_of_smell` | Perte d'odorat (0 = non, 1 = oui) |
| `spo2` | Saturation en oxygène (%) |
| `wbc` | Globules blancs (10^9/L) |
| `crp` | Protéine C-réactive (mg/L) |
| `infected` | Cible : 1 = infecté, 0 = sain |

## Résultats (jeu de test : 120 patients)

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Régression logistique | 0.825 | 0.811 | 0.796 | 0.804 |
| Réseau de neurones | 0.833 | 0.815 | 0.815 | 0.815 |

## Exercice d'application

### Question 1 — Classe `ClinicalPredictor`

Elle reçoit un modèle déjà entraîné, possède une méthode `diagnose(patient_data)` et retourne `"Infecté"` si la prédiction est ≥ 0.5, sinon `"Sain"` (voir `app/api.py`).

```python
class ClinicalPredictor:
    def __init__(self, model, scaler=None, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def diagnose(self, patient_data):
        x = np.asarray(patient_data, dtype=float).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        prediction = float(np.ravel(self.model.predict(x))[0])
        return "Infecté" if prediction >= self.threshold else "Sain"
```

### Question 2 — Quel concept de POO permet à `ClinicalPredictor` de fonctionner avec n'importe quel modèle ?

Le **polymorphisme** (via l'abstraction et le *duck typing*), combiné à l'**injection de dépendance**.

`ClinicalPredictor` ne connaît pas la classe concrète du modèle : il appelle seulement `predict()`. Que ce soit `LogisticRegression`, `NeuralNetwork` ou un futur modèle, tant que l'interface définie par `Model` est respectée, le code de `ClinicalPredictor` ne change pas. Le modèle est fourni de l'extérieur au lieu d'être créé à l'intérieur : le couplage est faible.

### Question 3 — Pourquoi séparer la prédiction (`ClinicalPredictor`) de l'entraînement (`Trainer`) ?

- **Responsabilité unique** : le `Trainer` apprend (lourd, long, avec les données d'entraînement) ; le `ClinicalPredictor` diagnostique (rapide, un patient à la fois).
- **Déploiement léger** : en production, on charge seulement un modèle entraîné, sans optimiseur ni jeu de données.
- **Sécurité et fiabilité** : l'application clinique ne peut pas modifier ou ré-entraîner le modèle par erreur ; le comportement reste stable et reproductible.
- **Évolutivité et testabilité** : on peut changer de modèle ou ré-entraîner sans toucher à l'application, et tester chaque partie isolément (avec un faux modèle pour le prédicteur).
- **Éviter la fuite de données** : le scaler appris sur le jeu d'entraînement est réutilisé tel quel à la prédiction.