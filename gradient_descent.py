
import numpy as np

class GradientDescentRegressor:
    def __init__(
        self,
        learning_rate=0.001,
        n_epochs=5000,
        batch_size=32,
        l2_lambda=0.0,
        early_stopping=True,
        tol=1e-6
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.tol = tol

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, n_epochs=None):
        """Fit model from scratch"""
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0.0
        self.loss_history = []

        epochs = self.n_epochs if n_epochs is None else n_epochs
        prev_loss = float("inf")

        for epoch in range(epochs):
            self._run_epoch(X, y)
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            if self.early_stopping and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

    def partial_fit(self, X, y):
        """Update model by one epoch (useful for learning curves)"""
        if self.weights is None or self.bias is None:
            # Initialize weights if not done yet
            self.weights = np.zeros((X.shape[1], 1))
            self.bias = 0.0
            self.loss_history = []

        self._run_epoch(X, y)
        loss = self._compute_loss(X, y)
        self.loss_history.append(loss)

    def _run_epoch(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            self._update_weights(X_batch, y_batch)

    def _update_weights(self, X, y):
        n = X.shape[0]
        y_pred = X @ self.weights + self.bias
        dw = (1 / n) * X.T @ (y_pred - y) + 2 * self.l2_lambda * self.weights
        db = (1 / n) * np.sum(y_pred - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        l2_penalty = self.l2_lambda * np.sum(self.weights ** 2)
        return mse + l2_penalty

    def predict(self, X):
        return X @ self.weights + self.bias
