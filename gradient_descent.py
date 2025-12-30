# import numpy as np


# class GradientDescentRegressor:
#     def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=None):
#         self.learning_rate = learning_rate
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size  # None for batch, 1 for stochastic, >1 for mini-batch
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros((n_features, 1))
#         self.bias = 0
        
#         for epoch in range(self.n_epochs):
#             if self.batch_size is None:  # Batch Gradient Descent
#                 X_batch = X
#                 y_batch = y
#             elif self.batch_size == 1:  # Stochastic Gradient Descent
#                 for i in range(n_samples):
#                     xi = X[i].reshape(1, -1)
#                     yi = y[i].reshape(1, -1)
#                     self._update_weights(xi, yi)
#                 continue
#             else:  # Mini-Batch Gradient Descent
#                 indices = np.random.permutation(n_samples)
#                 X_shuffled = X[indices]
#                 y_shuffled = y[indices]
#                 for start in range(0, n_samples, self.batch_size):
#                     end = start + self.batch_size
#                     X_batch = X_shuffled[start:end]
#                     y_batch = y_shuffled[start:end]
#                     self._update_weights(X_batch, y_batch)
#                 continue

#             # Update weights for batch gradient descent
#             self._update_weights(X_batch, y_batch)

#     def _update_weights(self, X, y):
#         n_samples = X.shape[0]
#         y_pred = X @ self.weights + self.bias
#         dw = (1/n_samples) * X.T @ (y_pred - y)
#         db = (1/n_samples) * np.sum(y_pred - y)
#         self.weights -= self.learning_rate * dw
#         self.bias -= self.learning_rate * db

#     def predict(self, X):
#         return X @ self.weights + self.bias

# import numpy as np


# class GradientDescentRegressor:
#     def __init__(
#         self,
#         learning_rate=0.01,
#         n_epochs=1000,
#         batch_size=None,
#         l2_lambda=0.0
#     ):
#         self.learning_rate = learning_rate
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size  # None=batch, 1=SGD, >1=mini-batch
#         self.l2_lambda = l2_lambda
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         n_samples, n_features = X.shape

#         self.weights = np.zeros((n_features, 1))
#         self.bias = 0.0

#         for epoch in range(self.n_epochs):

#             if self.batch_size is None:  # Batch GD
#                 self._update_weights(X, y)

#             elif self.batch_size == 1:  # SGD
#                 indices = np.random.permutation(n_samples)
#                 for i in indices:
#                     xi = X[i].reshape(1, -1)
#                     yi = y[i].reshape(1, -1)
#                     self._update_weights(xi, yi)

#             else:  # Mini-Batch GD
#                 indices = np.random.permutation(n_samples)
#                 X_shuffled = X[indices]
#                 y_shuffled = y[indices]

#                 for start in range(0, n_samples, self.batch_size):
#                     end = start + self.batch_size
#                     X_batch = X_shuffled[start:end]
#                     y_batch = y_shuffled[start:end]
#                     self._update_weights(X_batch, y_batch)

#     def _update_weights(self, X, y):
#         m = X.shape[0]

#         y_pred = X @ self.weights + self.bias
#         error = y_pred - y

#         # Gradient MSE
#         dw = (1 / m) * (X.T @ error)
#         db = (1 / m) * np.sum(error)

#         #  L2 Regularization (NE PAS pénaliser le biais)
#         if self.l2_lambda > 0:
#             dw += (self.l2_lambda / m) * self.weights

#         # Update
#         self.weights -= self.learning_rate * dw
#         self.bias -= self.learning_rate * db

#     def predict(self, X):
#         return X @ self.weights + self.bias



# import numpy as np


# class GradientDescentRegressor:
#     def __init__(
#         self,
#         learning_rate=0.001,
#         n_epochs=5000,
#         batch_size=32,
#         l2_lambda=0.0,
#         early_stopping=True,
#         tol=1e-6
#     ):
#         self.learning_rate = learning_rate
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#         self.l2_lambda = l2_lambda
#         self.early_stopping = early_stopping
#         self.tol = tol

#         self.weights = None
#         self.bias = None
#         self.loss_history = []

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros((n_features, 1))
#         self.bias = 0.0

#         prev_loss = float("inf")

#         for epoch in range(self.n_epochs):
#             indices = np.random.permutation(n_samples)
#             X_shuffled = X[indices]
#             y_shuffled = y[indices]

#             for start in range(0, n_samples, self.batch_size):
#                 end = start + self.batch_size
#                 X_batch = X_shuffled[start:end]
#                 y_batch = y_shuffled[start:end]
#                 self._update_weights(X_batch, y_batch)

#             # Loss (MSE + L2)
#             y_pred = self.predict(X)
#             mse = np.mean((y - y_pred) ** 2)
#             l2_penalty = self.l2_lambda * np.sum(self.weights ** 2)
#             loss = mse + l2_penalty
#             self.loss_history.append(loss)

#             if self.early_stopping and abs(prev_loss - loss) < self.tol:
#                 break

#             prev_loss = loss

#     def _update_weights(self, X, y):
#         n = X.shape[0]
#         y_pred = X @ self.weights + self.bias

#         dw = (1 / n) * X.T @ (y_pred - y) + 2 * self.l2_lambda * self.weights
#         db = (1 / n) * np.sum(y_pred - y)

#         self.weights -= self.learning_rate * dw
#         self.bias -= self.learning_rate * db

#     def predict(self, X):
#         return X @ self.weights + self.bias




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
