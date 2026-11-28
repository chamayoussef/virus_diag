"""Tests automatiques de virus_diag (lancés par GitHub Actions à chaque push)."""
from pathlib import Path

import numpy as np
import pytest

from app import ClinicalPredictor, diagnose_endpoint, FEATURES
from core import Dataset, LogisticRegression, NeuralNetwork
from pipeline import Trainer, Evaluator
from utils import Standardizer, accuracy, precision, recall, f1_score, confusion

DATA = Path(__file__).parent.parent / "data" / "patient_data.csv"


class FakeModel:
    """Faux modèle : prouve que ClinicalPredictor marche avec n'importe quel predict()."""

    def __init__(self, proba):
        self.proba = proba

    def predict(self, X):
        return np.array([self.proba])


# ---------- Dataset ----------
def test_dataset_split_sizes():
    ds = Dataset(DATA)
    X_train, X_test, y_train, y_test = ds.split()
    assert len(ds) == 600
    assert X_train.shape == (480, 9)
    assert X_test.shape == (120, 9)
    assert len(y_train) == 480 and len(y_test) == 120


# ---------- Utils ----------
def test_standardizer_mean_zero_std_one():
    X = np.random.default_rng(0).normal(10, 3, (100, 4))
    Xs = Standardizer().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-9)


def test_metrics_known_example():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1])
    assert confusion(y_true, y_pred) == (2, 1, 1, 1)
    assert accuracy(y_true, y_pred) == pytest.approx(0.6)
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)


# ---------- Modèles ----------
@pytest.mark.parametrize("model_cls", [LogisticRegression, NeuralNetwork])
def test_model_beats_random_guess(model_cls):
    ds = Dataset(DATA)
    X_train, X_test, y_train, y_test = ds.split()
    model = model_cls()
    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    scores = Evaluator(model, trainer.scaler).evaluate(X_test, y_test)
    assert scores["accuracy"] > 0.75


@pytest.mark.parametrize("model_cls", [LogisticRegression, NeuralNetwork])
def test_predict_returns_probabilities(model_cls):
    ds = Dataset(DATA)
    X_train, X_test, y_train, _ = ds.split()
    model = model_cls()
    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    proba = model.predict(trainer.scaler.transform(X_test))
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------- ClinicalPredictor ----------
def test_predictor_infected_when_proba_above_threshold():
    assert ClinicalPredictor(FakeModel(0.9)).diagnose([0] * 9) == "Infecté"


def test_predictor_healthy_when_proba_below_threshold():
    assert ClinicalPredictor(FakeModel(0.1)).diagnose([0] * 9) == "Sain"


def test_predictor_threshold_is_inclusive():
    assert ClinicalPredictor(FakeModel(0.5)).diagnose([0] * 9) == "Infecté"


def test_predictor_works_with_real_model():
    ds = Dataset(DATA)
    X_train, _, y_train, _ = ds.split()
    model = LogisticRegression()
    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    predictor = ClinicalPredictor(model, trainer.scaler)
    assert predictor.diagnose([60, 39.0, 1, 1, 1, 1, 91.0, 4.8, 55.0]) == "Infecté"
    assert predictor.diagnose([30, 36.8, 0, 0, 0, 0, 98.0, 7.0, 4.0]) == "Sain"


# ---------- API ----------
def test_endpoint_rejects_missing_fields():
    result = diagnose_endpoint(None, {"age": 40})
    assert result["status"] == "error"
    assert "temperature" in result["missing_fields"]


def test_endpoint_returns_diagnosis():
    payload = dict(zip(FEATURES, [30, 36.8, 0, 0, 0, 0, 98.0, 7.0, 4.0]))
    result = diagnose_endpoint(ClinicalPredictor(FakeModel(0.1)), payload)
    assert result == {"status": "ok", "diagnosis": "Sain"}