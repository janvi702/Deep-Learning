import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Dataset Simulation (Sentiment Analysis Task) ---
num_samples = 1200
vocab_size = 5000
embed_dim = 64
hidden_dim = 128
output_size = 2  # Binary classification: Negative (0) or Positive (1)
seq_len = 25

# Generate random word indices and binary sentiment labels
X = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
y = np.random.randint(0, 2, size=(num_samples,))

# Train/Test Split (80% train, 20% test)
train_size = int(0.8 * num_samples)
X_train, X_test = torch.tensor(X[:train_size], dtype=torch.long), torch.tensor(X[train_size:], dtype=torch.long)
y_train, y_test = torch.tensor(y[:train_size], dtype=torch.long), torch.tensor(y[train_size:], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# --- 2. Configurable RNN Architecture ---
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size, rnn_type='lstm', num_layers=1):
        super(SentimentRNN, self).__init__()
        
        # Embedding layer to convert sparse token indices into dense representations
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Select the RNN cell variant
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Unsupported rnn_type. Choose from 'rnn', 'lstm', or 'gru'.")
            
        # Fully connected classification head
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) # Shape: (batch_size, seq_len, embed_dim)
        
        # Forward pass through the chosen RNN variant
        if self.rnn_type == 'lstm':
            out, (hidden, cell) = self.rnn(embedded)
        else:
            out, hidden = self.rnn(embedded)
            
        # Extract the output from the final time step for classification: (batch_size, hidden_dim)
        last_timestep_out = out[:, -1, :]
        
        # Compute classification logits
        logits = self.fc(last_timestep_out)
        return logits

# --- 3. Training and Evaluation Routine ---
def train_and_evaluate(rnn_type='lstm', epochs=3):
    print(f"\n==============================")
    print(f" Training Model: {rnn_type.upper()} ")
    print(f"==============================")
    
    model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_size, rnn_type=rnn_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss / len(train_loader):.4f}")
        
    # Evaluation Phase (Runs after training is complete)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n--- {rnn_type.upper()} Performance Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

# --- 4. Experimenting with Different RNN Variants ---
variants = ['rnn', 'lstm', 'gru']
for variant in variants:
    train_and_evaluate(rnn_type=variant, epochs=3)
