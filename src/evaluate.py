import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_PATH = "model/image_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)

classes = checkpoint["classes"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder("data/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

accuracy = accuracy_score(y_true, y_pred)

precision, recall, _, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)

cm = confusion_matrix(y_true, y_pred)

metrics = {
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "confusion_matrix": cm.tolist()
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation completed. Metrics saved to results/metrics.json")
