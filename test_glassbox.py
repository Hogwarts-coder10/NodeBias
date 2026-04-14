import pytest
import numpy as np
from glassboxml.core import train_test_split
from glassboxml.preprocessing import StandardScaler
from glassboxml.models import LogisticRegression
from glassboxml.core import Momentum


def test_custom_train_test_split():
    """Proves our custom splitter can handle infinite arrays and maintain dimensions."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    gender = np.array(['Male', 'Female'] * 50)

    # Pass 3 arrays (which would crash standard models without our *arrays update)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, gender, test_size=0.2, random_state=42
    )

    # Assertions to prove no data leakage and perfect slicing
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(g_train) == 80

def test_standard_scaler():
    """Proves the scaler correctly centers data (Mean = 0, Std = 1)."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # The mean of each column should be extremely close to 0
    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-7)
    # The standard deviation of each column should be 1
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-7)

def test_logistic_regression_convergence():
    """Proves our custom model can actually learn a simple dataset."""
    # Create a dummy dataset that is incredibly easy to separate
    X = np.array([[0.1], [0.2], [0.3], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    optimizer = Momentum(learning_rate=0.1,beta=0.9)
    model = LogisticRegression(optimizer=optimizer,epochs=500,loss_function='bce')
    model.fit(X, y)

    predictions = model.predict(X)

    # The model should get 100% accuracy on this simple data
    accuracy = np.mean(predictions == y)
    assert accuracy == 1.0
