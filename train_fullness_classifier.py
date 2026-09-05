"""
train_fullness_classifier.py

Image classification untuk basket fullness (empty/light/medium/full)
pakai TRANSFER LEARNING dari MobileNetV2 pretrained (ImageNet).

Kenapa MobileNetV2 + transfer learning?
- Cocok untuk dataset kecil (80-150 gambar cukup), karena cuma melatih
  ulang layer terakhir, bukan seluruh network dari nol.
- Ringan & cepat, jalan di CPU biasa (gak wajib GPU) -- penting untuk demo
  hackathon di laptop.

CARA PAKAI:
1. Install dependency (di laptop/Colab, BUKAN di sini):
       pip install torch torchvision pillow

2. Susun folder foto seperti ini:
       dataset/
         train/
           empty/   -> foto-foto basket kosong
           light/   -> foto-foto isi sedikit
           medium/  -> foto-foto isi sedang
           full/    -> foto-foto isi penuh
         val/
           empty/
           light/
           medium/
           full/
   (val = subset kecil buat validasi, ambil ~20% dari total foto per kelas)

3. Jalankan:
       python train_fullness_classifier.py --data_dir ./dataset --epochs 15

4. Model hasil training tersimpan di fullness_classifier.pt
   Pakai predict.py (file terpisah) untuk inference ke foto baru.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASS_NAMES = ["empty", "light", "medium", "full"]  # urutan harus konsisten


def build_dataloaders(data_dir: str, batch_size: int = 16):
    # Augmentasi ringan untuk train (dataset kecil -> augmentasi bantu generalisasi)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    # Pastikan urutan kelas sesuai CLASS_NAMES (ImageFolder sort alfabetis otomatis)
    print(f"Kelas terdeteksi: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze semua layer feature extractor (gak dilatih ulang)
    for param in model.features.parameters():
        param.requires_grad = False

    # Ganti classifier terakhir sesuai jumlah kelas kita (4 kelas fullness)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


def train(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    # Hanya optimize parameter yang requires_grad=True (classifier terakhir)
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
            print(f"  -> model tersimpan (val_acc terbaik: {val_acc:.3f})")

    print(f"\nTraining selesai. Best val accuracy: {best_val_acc:.3f}")
    print("Model tersimpan di: fullness_classifier.pt")


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
                         help="Path ke folder dataset (berisi train/ dan val/)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    train_loader, val_loader, classes = build_dataloaders(args.data_dir, args.batch_size)
    model = build_model(num_classes=len(classes))

    train(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Simpan mapping kelas juga, supaya predict.py tahu urutan class index -> nama
    with open("class_names.txt", "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    main()
