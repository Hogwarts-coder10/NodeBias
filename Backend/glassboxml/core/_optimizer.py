import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Base class for all optimization algorithms in GlassBoxML.
    """

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    @abstractmethod
    def update(self, coef, intercept, dw, db):
        """
        Updates and returns the new coefficients and intercept based on the gradients.
        """

        pass

class GradientDescent(Optimizer):
    """
    Standard Batch Gradient Descent.
    """

    def update(self, coef, intercept, dw, db):
        # The classic update rule: θ = θ - α∇J(θ)
        coef -= self.lr * dw
        intercept -= self.lr * db
        return coef, intercept

class Momentum(Optimizer):
    """
    Gradient Descent with Momentum to accelerate learning.
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_dw = None
        self.v_db = None

    def update(self, coef, intercept, dw, db):
        # Initialize velocities on the first step
        if self.v_dw is None:
            self.v_dw = np.zeros_like(coef)
            self.v_db = 0.0
        
        # Exponential moving average of gradients: v = βv + (1-β)∇J
        self.v_dw = self.beta * self.v_dw + (1 - self.beta) * dw
        self.v_db = self.beta * self.v_db + (1 - self.beta) * db
        
        # Apply the updates
        coef -= self.lr * self.v_dw
        intercept -= self.lr * self.v_db
        
        return coef, intercept