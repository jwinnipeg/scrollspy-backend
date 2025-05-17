import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Load Hugging Face token from environment
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Load the model
model_name = "Ateeqq/ai-vs-human-image-detector"
extractor = AutoFeatureExtractor.from_pretrained(model_name, token=hf_token)
model = AutoModelForImageClassification.from_pretrained(model_name, token=hf_token)

# Image classification function
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
