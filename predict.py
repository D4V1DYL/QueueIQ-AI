"""
predict.py

Inference script untuk basket fullness classifier yang sudah dilatih
(hasil dari train_fullness_classifier.py).

CARA PAKAI:
    python predict.py --image path/ke/foto_basket.jpg
    python predict.py --image path/ke/folder_foto/   (proses semua foto di folder)

Output: prediksi kelas (empty/light/medium/full) + confidence score per gambar.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

MODEL_PATH = "fullness_classifier.pt"
CLASS_NAMES_PATH = "class_names.txt"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(num_classes: int, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single(model, image_path, class_names, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return class_names[pred_idx], probs[pred_idx].item(), probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                         help="Path ke satu file gambar, atau folder berisi banyak gambar")
    args = parser.parse_args()

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH} tidak ditemukan. Jalankan train_fullness_classifier.py dulu."
        )
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes=len(class_names), device=device)

    if os.path.isdir(args.image):
        image_paths = [
            os.path.join(args.image, fname)
            for fname in sorted(os.listdir(args.image))
            if fname.lower().endswith(IMAGE_EXTENSIONS)
        ]
    else:
        image_paths = [args.image]

    for path in image_paths:
        label, confidence, all_probs = predict_single(model, path, class_names, device)
        prob_str = ", ".join(f"{c}={p:.2f}" for c, p in zip(class_names, all_probs))
        print(f"{os.path.basename(path):30s} -> {label:8s} (confidence={confidence:.2f}) | {prob_str}")


if __name__ == "__main__":
    main()
