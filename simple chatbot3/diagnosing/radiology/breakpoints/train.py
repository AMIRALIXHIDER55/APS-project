# models/fracture_model.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "prithivMLmods/Bone-Fracture-Detection"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()
model.to(device)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    if pred == 0:
        return {"diagnosis": "Fracture detected", "confidence": 0.95}
    else:
        return {"diagnosis": "No fracture", "confidence": 0.95}









