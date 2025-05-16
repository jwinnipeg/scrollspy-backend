from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the AI image detection model
extractor = AutoFeatureExtractor.from_pretrained("guyfloki/ai-image-detector")
model = AutoModelForImageClassification.from_pretrained("guyfloki/ai-image-detector")

def classify_image(image: Image.Image) -> dict:
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    confidence, predicted_class = torch.max(probabilities, dim=0)
    label = model.config.id2label[predicted_class.item()]
    return {
        "label": label,
        "confidence": round(confidence.item(), 4)
    }

