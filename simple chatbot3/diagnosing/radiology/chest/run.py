# models/chest_model.py
import torch
from torchvision import models, transforms
from PIL import Image

CHEXNET_CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural Thickening","Hernia"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_chest_model(ckpt_path):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Sequential(torch.nn.Linear(num_ftrs, len(CHEXNET_CLASSES)))
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"]
    new_state = {k.replace("densenet121.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model.to(device)
    return model

# ⚠️ مسیر مدل خود را اینجا بگذارید
chest_model_path = r"C:\Users\BEROOZDG\Desktop\diagnosing\radiology\chest\m-30012020-104001.pth.tar"
model = load_chest_model(chest_model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out).cpu().numpy()[0]
    results = sorted(zip(CHEXNET_CLASSES, probs), key=lambda x: x[1], reverse=True)
    for cls, p in results:
        if p > 0.5:
            return {"diagnosis": cls, "confidence": float(p)}
    return {"diagnosis": "Normal", "confidence": 1.0 - float(max(probs))}


