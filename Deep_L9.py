import numpy as np
import matplotlib.pyplot as plt

# Sample data
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Matrix Addition
matrix_add = A + B

# 2. Matrix Multiplication (Dot Product)
matrix_mul = A @ B  # equivalent to np.dot(A, B)

# 3. Element-wise Operations (e.g., squaring each element, exponential)
element_wise_sq = np.square(A)
element_wise_exp = np.exp(A)

print("Matrix Addition:\n", matrix_add)
print("\nMatrix Multiplication:\n", matrix_mul)
print("\nElement-wise Square:\n", element_wise_sq)

# --- Activation Functions & Derivatives ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(sig):
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# --- Synthetic Dataset (XOR-like problem) ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])  # Target outputs

# --- Network Initialization ---
np.random.seed(42)
input_dim = 2
hidden_dim = 3
output_dim = 1

# Weights and Biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

# Hyperparameters & Tracking lists for visualization
lr = 0.1
epochs = 1000
loss_history = []
grad_W1_history = []
grad_W2_history = []
hidden_activations_history = []

for epoch in range(epochs):
    # ==================== FEEDFORWARD ====================
    Z1 = X @ W1 + b1          # Linear step (Hidden)
    A1 = relu(Z1)             # Activation step (Hidden ReLU)
    Z2 = A1 @ W2 + b2         # Linear step (Output)
    A2 = sigmoid(Z2)          # Activation step (Output Sigmoid)
    
    # Compute Binary Cross-Entropy Loss
    loss = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
    loss_history.append(loss)
    
    # Track intermediate activations & outputs on the last epoch
    if epoch == epochs - 1:
        hidden_activations_history.append(A1.copy())
        
    # ==================== BACKPROPAGATION ====================
    # Error at output layer
    dZ2 = A2 - Y                  
    
    # Gradients for layer 2
    dW2 = A1.T @ dZ2              
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Error propagated to hidden layer
    dA1 = dZ2 @ W2.T              
    dZ1 = dA1 * relu_derivative(Z1)
    
    # Gradients for layer 1
    dW1 = X.T @ dZ1               
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Track gradients magnitudes over time
    grad_W1_history.append(np.linalg.norm(dW1))
    grad_W2_history.append(np.linalg.norm(dW2))
    
    # ==================== PARAMETER UPDATE ====================
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print("Final Predictions after training:\n", np.round(A2, 4))

# ==================== VISUALIZATION ====================
plt.figure(figsize=(14, 4))

# Plot 1: Loss Curve
plt.subplot(1, 3, 1)
plt.plot(loss_history, color='crimson', lw=2)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Gradient Norms (Flow of Backpropagation)
plt.subplot(1, 3, 2)
plt.plot(grad_W1_history, label='||dW1|| (Input-Hidden)', color='dodgerblue')
plt.plot(grad_W2_history, label='||dW2|| (Hidden-Output)', color='darkorange')
plt.title("Gradient Magnitudes During Backprop")
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 3: Hidden Layer Activations for Each Sample (Final Epoch)
plt.subplot(1, 3, 3)
plt.imshow(hidden_activations_history[0], cmap='viridis', aspect='auto')
plt.colorbar(label='Activation Value')
plt.title("Hidden Layer Outputs ($A_1$)\n(Per Sample)")
plt.xlabel("Hidden Neurons")
plt.ylabel("Samples")
plt.xticks([0, 1, 2], ['Neuron 1', 'Neuron 2', 'Neuron 3'])
plt.yticks([0, 1, 2, 3], ['Sample 0', 'Sample 1', 'Sample 2', 'Sample 3'])

plt.tight_layout()
plt.show()
