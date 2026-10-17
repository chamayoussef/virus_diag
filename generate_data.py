"""Génère des données patients synthétiques (exercice pédagogique)."""
import csv
from pathlib import Path

import numpy as np

N = 600
OUT = Path(__file__).parent / "data" / "patient_data.csv"
HEADER = ["age", "temperature", "cough", "fatigue", "shortness_of_breath",
          "loss_of_smell", "spo2", "wbc", "crp", "infected"]

rng = np.random.default_rng(42)
infected = rng.random(N) < 0.45
rows = []

for inf in infected:
    age = int(np.clip(rng.normal(45, 15), 1, 95))
    if inf:
        temp = rng.normal(38.0, 0.9); cough = rng.random() < 0.70
        fatigue = rng.random() < 0.65; breath = rng.random() < 0.35
        smell = rng.random() < 0.40; spo2 = rng.normal(95.0, 2.5)
        wbc = rng.normal(6.0, 2.0); crp = rng.normal(30, 22)
    else:
        temp = rng.normal(37.3, 0.8); cough = rng.random() < 0.30
        fatigue = rng.random() < 0.35; breath = rng.random() < 0.12
        smell = rng.random() < 0.10; spo2 = rng.normal(96.8, 2.0)
        wbc = rng.normal(6.8, 2.0); crp = rng.normal(14, 12)
    rows.append([age, round(temp, 1), int(cough), int(fatigue), int(breath),
                 int(smell), round(min(spo2, 100), 1),
                 round(max(wbc, 1.5), 1), round(max(crp, 0.5), 1), int(inf)])

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print(f"{len(rows)} patients écrits dans {OUT}")