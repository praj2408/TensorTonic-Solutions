import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    
    Parameters:
        X : shape (n_samples, n_features)
        y : shape (n_samples,)
        lr : learning rate
        steps : number of gradient descent iterations
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # Linear model
        z = X @ w + b
        
        # Prediction
        y_hat = _sigmoid(z)
        
        # Gradients (binary cross-entropy loss)
        dw = (1 / n_samples) * (X.T @ (y_hat - y))
        db = (1 / n_samples) * np.sum(y_hat - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b