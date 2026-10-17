"""Métriques de classification binaire."""


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())   # vrais positifs
    tn = int(((y_true == 0) & (y_pred == 0)).sum())   # vrais négatifs
    fp = int(((y_true == 0) & (y_pred == 1)).sum())   # faux positifs
    fn = int(((y_true == 1) & (y_pred == 0)).sum())   # faux négatifs
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred):
    tp, _, _, fn = confusion(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0