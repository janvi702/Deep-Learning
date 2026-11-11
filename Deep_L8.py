import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data for Linear Regression
# True relationship: y = 3x + 4 + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1) * 0.5

# 2. Define Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m_samples = len(X)
    
    # Initialize parameters (weights and bias) randomly or to zero
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    cost_history = []
    
    for i in range(iterations):
        # Forward pass: predictions
        y_pred = w * X + b
        
        # Calculate Mean Squared Error (MSE) Cost
        cost = (1 / (2 * m_samples)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # Backward pass: compute gradients
        dw = (1 / m_samples) * np.sum((y_pred - y) * X)
        db = (1 / m_samples) * np.sum(y_pred - y)
        
        # Update parameters using gradients and learning rate
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, w = {w[0]:.4f}, b = {b[0]:.4f}")
            
    # FIXED: Return statement moved outside the for loop
    return w, b, cost_history

# 3. Run Optimization
learning_rate = 0.1
iterations = 500
optimal_w, optimal_b, cost_history = gradient_descent(X, y, learning_rate, iterations)

print("\n--- Optimization Complete ---")
print(f"Optimal Weight (w): {optimal_w[0]:.4f}")
print(f"Optimal Bias (b): {optimal_b[0]:.4f}")

# 4. Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Cost Function Convergence
plt.subplot(1, 2, 1)
plt.plot(range(iterations), cost_history, color='b', lw=2)
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (Cost)")
plt.grid(True)

# Plot 2: Data Fit
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='gray', alpha=0.7, label='Data points')
plt.plot(X, optimal_w * X + optimal_b, color='r', lw=2, label='Fitted Line')
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
