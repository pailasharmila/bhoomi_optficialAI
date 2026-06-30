from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import io

MODEL_NAME = "mesabo/agri-plant-disease-resnet50"

# Load once at import time — not per request
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

def parse_label(raw_label: str) -> dict:
    """
    Raw label looks like: 'Tomato___Early_blight'
    Returns: { "crop": "Tomato", "disease": "Early blight" }
    """
    parts = raw_label.replace("___", "||").replace("_", " ").split("||")
    crop = parts[0].strip() if len(parts) > 0 else "Unknown"
    disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()
    return {
        "crop": crop,
        "disease": "Healthy" if is_healthy else disease,
        "is_healthy": is_healthy
    }

def predict_disease(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = probs.argmax(-1).item()
        confidence = probs[0][predicted_idx].item()

    raw_label = model.config.id2label[predicted_idx]
    parsed = parse_label(raw_label)

    return {
        **parsed,
        "confidence": round(confidence * 100, 2),
        "raw_label": raw_label
    }