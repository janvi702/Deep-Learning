# %%
# Import required libraries
import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
# %%
# Load Wine Dataset
data = load_wine()
X_full = data.data
y_full = data.target

# --- 1. Classification Setup (Binary: Class 0 vs Class 1) ---
binary_mask = y_full < 2
X_clf = X_full[binary_mask]
y_clf = y_full[binary_mask]

# Split and Scale Classification Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)
scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

# --- 2. Regression Setup (Predicting 'alcohol' feature index 0) ---
# Use all features except index 0 to predict index 0
X_reg = np.delete(X_full, 0, axis=1)
y_reg = X_full[:, 0]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

print(f"Classification Train Shape: {X_train_c.shape}, Test Shape: {X_test_c.shape}")
print(f"Regression Train Shape: {X_train_r.shape}, Test Shape: {X_test_r.shape}")
# %%
class ScratchLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
      # %%
class ScratchLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
      # %%
class ScratchKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances between x and all training samples
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and get indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract labels of the k-nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common class label
        vals, counts = np.unique(k_nearest_labels, return_counts=True)
        return vals[np.argmax(counts)]
      # %%
results = []

# --- 1. Linear Regression Evaluation ---
start_time = time.time()
lin_reg = ScratchLinearRegression(learning_rate=0.01, n_iterations=1000)
lin_reg.fit(X_train_r, y_train_r)
train_time_lin = time.time() - start_time

start_time = time.time()
y_pred_lin = lin_reg.predict(X_test_r)
inf_time_lin = time.time() - start_time
mse = mean_squared_error(y_test_r, y_pred_lin)

results.append({
    "Algorithm": "Linear Regression",
    "Task": "Regression",
    "Train Time (s)": f"{train_time_lin:.5f}",
    "Inference Time (s)": f"{inf_time_lin:.5f}",
    "Performance Metric": f"MSE: {mse:.4f}"
})

# --- 2. Logistic Regression Evaluation ---
start_time = time.time()
log_reg = ScratchLogisticRegression(learning_rate=0.01, n_iterations=1000)
log_reg.fit(X_train_c, y_train_c)
train_time_log = time.time() - start_time

start_time = time.time()
y_pred_log = log_reg.predict(X_test_c)
inf_time_log = time.time() - start_time
acc_log = accuracy_score(y_test_c, y_pred_log)

results.append({
    "Algorithm": "Logistic Regression",
    "Task": "Classification",
    "Train Time (s)": f"{train_time_log:.5f}",
    "Inference Time (s)": f"{inf_time_log:.5f}",
    "Performance Metric": f"Accuracy: {acc_log * 100:.2f}%"
})

# --- 3. K-Nearest Neighbors Evaluation ---
start_time = time.time()
knn = ScratchKNN(k=3)
knn.fit(X_train_c, y_train_c)
train_time_knn = time.time() - start_time  # Lazy learner (instant fit)

start_time = time.time()
y_pred_knn = knn.predict(X_test_c)
inf_time_knn = time.time() - start_time
acc_knn = accuracy_score(y_test_c, y_pred_knn)

results.append({
    "Algorithm": "K-Nearest Neighbors (k=3)",
    "Task": "Classification",
    "Train Time (s)": f"{train_time_knn:.5f}",
    "Inference Time (s)": f"{inf_time_knn:.5f}",
    "Performance Metric": f"Accuracy: {acc_knn * 100:.2f}%"
})

# --- Display Comparison Table ---
import pandas as pd
df_results = pd.DataFrame(results)
display(df_results)
