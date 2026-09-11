import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from model import load_model

DATA_DIR = "data/processed/val"
CHECKPOINT_PATH = "model/checkpoints/defect_model.pt"

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(DATA_DIR, transform=tf)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model, class_names = load_model(CHECKPOINT_PATH, num_classes=len(val_ds.classes), device=device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()