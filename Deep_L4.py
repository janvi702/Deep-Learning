# %% [import-libraries]
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# %% [load-and-preprocess-data]
# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}, Number of target classes: {len(np.unique(y))}")
# %% [define-model]
class WineClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn):
        super(WineClassifierNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Construct multiple hidden layers with the specified activation function
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn)
            prev_dim = h_dim

        # Final output layer (raw logits for CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# %% [training-evaluation-functions]
def train_model(activation_fn, epochs=60, lr=0.01):
    input_dim = X_train.shape[1]
    hidden_dims = [32, 16]  # Multiple hidden layers architecture
    output_dim = len(np.unique(y))

    model = WineClassifierNN(input_dim, hidden_dims, output_dim, activation_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        test_accuracies.append(acc)

    end_time = time.time()
    training_time = end_time - start_time

    return model, train_losses, test_accuracies, training_time
  # %% [run-experiments]
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'Leaky ReLU': nn.LeakyReLU(negative_slope=0.01)
}

results = {}

for name, act_fn in activations.items():
    print(f"Training network using {name} activation function...")
    model, losses, accuracies, train_time = train_model(act_fn, epochs=60, lr=0.01)
    results[name] = {
        'losses': losses,
        'accuracies': accuracies,
        'time': train_time,
        'final_accuracy': accuracies[-1]
    }
print("Training completed for all configurations.")
# %% [visualization-and-comparison]
plt.figure(figsize=(14, 5))

# Plot 1: Convergence (Training Loss)
plt.subplot(1, 2, 1)
for name, res in results.items():
    plt.plot(res['losses'], label=name, linewidth=2)
plt.title('Convergence Comparison: Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)

# Plot 2: Accuracy over Epochs
plt.subplot(1, 2, 2)
for name, res in results.items():
    plt.plot(res['accuracies'], label=name, linewidth=2)
plt.title('Accuracy Comparison: Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Summary Table Generation
summary_data = []
for name, res in results.items():
    summary_data.append({
        'Activation Function': name,
        'Final Test Accuracy (%)': f"{res['final_accuracy'] * 100:.2f}%",
        'Training Time (Seconds)': f"{res['time']:.4f}s",
        'Final Cross-Entropy Loss': f"{res['losses'][-1]:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n--- Performance Summary Table ---")
display(summary_df) if 'display' in globals() else print(summary_df)
