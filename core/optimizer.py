"""Optimiseur : descente de gradient simple."""


class GradientDescent:
    def __init__(self, lr=0.1):
        self.lr = lr   # learning rate : taille du pas

    def step(self, params, grads):
        """Met à jour chaque paramètre (liste de tableaux numpy) en place."""
        for p, g in zip(params, grads):
            p -= self.lr * g