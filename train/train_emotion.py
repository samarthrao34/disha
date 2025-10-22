import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import argparse
import os

# -----------------------------
# 🔧 Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Emotion Recognition Trainer (GPU Optimized)")
parser.add_argument('--data-dir', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--unfreeze-backbone', action='store_true', help='Unfreeze base CNN layers for fine-tuning')
args = parser.parse_args()

# -----------------------------
# 💻 Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Training on: {device}")
if torch.cuda.is_available():
    print(f"💪 GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not detected — training on CPU.")

# -----------------------------
# 🧠 Data Preparation
# -----------------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(args.data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

num_classes = len(train_dataset.classes)
print(f"📁 Found {len(train_dataset)} images across {num_classes} emotion classes: {train_dataset.classes}")

# -----------------------------
# 🧩 Model Setup (ResNet18)
# -----------------------------
model = models.resnet18(weights='IMAGENET1K_V1')
if not args.unfreeze_backbone:
    for param in model.parameters():
        param.requires_grad = False

# Replace classifier head
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -----------------------------
# ⚙️ Training Setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 🚀 Training Loop
# -----------------------------
print("\n🚀 Starting Training...\n")
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"📘 Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# -----------------------------
# 💾 Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "emotion_model_gpu.pth")
torch.save(model.state_dict(), model_path)
print(f"\n✅ Training Complete! Model saved to: {model_path}")
