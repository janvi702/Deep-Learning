import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Define Data Augmentation and Normalization
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load Pre-trained ResNet18 and Freeze Feature Extractor (Phase 1)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace Classifier Head for Custom Task (e.g., 5 classes)
num_classes = 5
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

# 4. Setup Loss and Optimizer (Training head only)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("--- Phase 1: Feature Extraction Head Initialized ---")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 5. Optional: Phase 2 Fine-Tuning (Unfreezing upper layers with a lower LR)
# Uncomment below to fine-tune the final ResNet block:
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

print("\n--- Phase 2: Fine-Tuning Upper Layers Enabled ---")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
