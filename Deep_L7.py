import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))} ({target_names})")

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoaders
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(X_train_tensor)}, Test samples: {len(X_test_tensor)}")
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FeedforwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y))

fnn_model = FeedforwardNN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes).to(device)
print(fnn_model)
class ConvNet1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # Reshape tabular vector to [batch_size, 1, num_features]
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

cnn_model = ConvNet1D(input_dim=input_dim, num_classes=num_classes).to(device)
print(cnn_model)
class RecurrentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RecurrentNN, self).__init__()
        # Treat sequence of 13 features as steps of size 1
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Reshape to [batch_size, seq_len=13, feature_dim=1]
        x = x.unsqueeze(-1)
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

rnn_model = RecurrentNN(input_dim=input_dim, hidden_dim=32, num_classes=num_classes).to(device)
print(rnn_model)
def train_and_evaluate(model, train_loader, test_loader, epochs=80, lr=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    model.train()

    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        train_losses.append(epoch_loss / len(train_loader.dataset))

    training_time = time.time() - start_time

    # Inference & Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    inference_start = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    inference_time = time.time() - inference_start
    accuracy = accuracy_score(all_targets, all_preds)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'accuracy': accuracy,
        'train_time': training_time,
        'inference_time': inference_time,
        'num_params': num_params,
        'preds': all_preds,
        'targets': all_targets,
        'losses': train_losses
    }
models = {
    'Feedforward NN (FNN)': FeedforwardNN(input_dim, 64, num_classes).to(device),
    '1D CNN': ConvNet1D(input_dim, num_classes).to(device),
    'Recurrent NN (LSTM)': RecurrentNN(input_dim, 32, num_classes).to(device)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    res = train_and_evaluate(model, train_loader, test_loader, epochs=80, lr=0.005)
    results[name] = res
    print(f"{name} -> Accuracy: {res['accuracy']*100:.2f}% | Train Time: {res['train_time']:.4f}s | Parameters: {res['num_params']}")
summary_data = []
for name, res in results.items():
    summary_data.append({
        'Model Architecture': name,
        'Test Accuracy (%)': res['accuracy'] * 100,
        'Training Time (s)': res['train_time'],
        'Inference Time (s)': res['inference_time'],
        'Trainable Parameters': res['num_params']
    })

comparison_df = pd.DataFrame(summary_data)
display(comparison_df)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss Curves
for name, res in results.items():
    axes[0, 0].plot(res['losses'], label=name)
axes[0, 0].set_title('Training Loss Curves')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Accuracy Comparison
sns.barplot(data=comparison_df, x='Model Architecture', y='Test Accuracy (%)', ax=axes[0, 1])
axes[0, 1].set_title('Test Accuracy (%)')
axes[0, 1].set_ylim(0, 105)

# 3. Training Time
sns.barplot(data=comparison_df, x='Model Architecture', y='Training Time (s)', ax=axes[1, 0])
axes[1, 0].set_title('Training Time (Seconds)')

# 4. Parameter Count
sns.barplot(data=comparison_df, x='Model Architecture', y='Trainable Parameters', ax=axes[1, 1])
axes[1, 1].set_title('Trainable Parameter Count')

plt.tight_layout()
plt.show()
