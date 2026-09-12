"""
train_fullness_classifier.py

Image classification for basket fullness (empty/light/medium/full/...)
using TRANSFER LEARNING from MobileNetV2 pretrained on ImageNet.

Why MobileNetV2 + transfer learning?
- Suits small datasets (80-150 images are enough) because only the last
  classifier layer is retrained (optionally the last few feature blocks),
  not the whole network from scratch.
- Light and fast, runs on a normal CPU (no GPU required) -- important for a
  hackathon demo on a laptop.

USAGE:
1. Install the dependencies:
       pip install torch torchvision pillow

2. Arrange the photos like this:
       dataset/
         train/
           empty/   -> photos of empty baskets
           light/   -> lightly filled
           medium/  -> half full
           full/    -> full
         val/
           empty/
           light/
           medium/
           full/
   (val = a small validation subset, about 20% of the photos per class)

3. Run:
       python train_fullness_classifier.py --data_dir ./dataset --epochs 15
   With noisy scraped data, partially unfreezing the backbone helps:
       python train_fullness_classifier.py --data_dir ./dataset --epochs 25 --lr 4e-4 --unfreeze_last 4

4. The trained model is saved to fullness_classifier.pt (+ class_names.txt).
   Use predict.py (separate file) for inference on new photos.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASS_NAMES = ["empty", "light", "medium", "full"]  # order must stay consistent


def build_dataloaders(data_dir: str, batch_size: int = 16):
    # Light augmentation for training (small dataset -> augmentation helps generalisation)
    train_transform = transforms.Compose([
        # random crop + small rotation: stock photos and overhead-camera crops differ
        # in scale, this augmentation makes the model more robust to that
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    # ImageFolder sorts classes alphabetically; the order is saved to class_names.txt
    print(f"Classes found: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int, unfreeze_last: int = 0):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the feature extractor; optionally unfreeze the last N blocks so the
    # features adapt to the basket domain (small dataset -> never all of them)
    for param in model.features.parameters():
        param.requires_grad = False
    if unfreeze_last > 0:
        for block in list(model.features)[-unfreeze_last:]:
            for param in block.parameters():
                param.requires_grad = True

    # Replace the final classifier with one sized for our fullness classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


def train(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # scraped labels are somewhat noisy
    # Only optimise parameters with requires_grad=True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fullness_classifier.pt")
            print(f"  -> model saved (best val_acc: {val_acc:.3f})")

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.3f}")
    print("Model saved to: fullness_classifier.pt")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the dataset folder (containing train/ and val/)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_last", type=int, default=0,
                        help="Number of trailing MobileNetV2 feature blocks to train too (default 0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = build_dataloaders(args.data_dir, args.batch_size)
    model = build_model(num_classes=len(classes), unfreeze_last=args.unfreeze_last)

    train(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Save the class mapping too, so predict.py knows class index -> name
    with open("class_names.txt", "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    main()
