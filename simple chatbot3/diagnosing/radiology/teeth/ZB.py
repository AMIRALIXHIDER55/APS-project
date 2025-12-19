# models/dental_model.py
from ultralytics import YOLO

CONF_THRESHOLD = 0.5

# ⚠️ مسیر مدل دندان خود را اینجا قرار دهید
MODEL_PATH = r"C:\Users\BEROOZDG\Desktop\diagnosing\radiology\teeth\best (1).pt"
model = YOLO(MODEL_PATH)

def predict(image_path):
    results = model.predict(source=image_path, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    detected_diseases = {}
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            disease_name = names[cls_id]
            detected_diseases[disease_name] = max(conf, detected_diseases.get(disease_name, 0))
    if detected_diseases:
        disease = max(detected_diseases, key=lambda k: detected_diseases[k])
        conf = detected_diseases[disease]
        return {"diagnosis": disease, "confidence": float(conf)}
    return {"diagnosis": "No dental issue detected", "confidence": 0.95}

