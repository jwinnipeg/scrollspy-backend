from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# ✅ Use the real AI model
model_id = "microsoft/beit-large-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

def classify_image(image: Image.Image):
    # Convert image and prepare input with padding
    inputs = processor(images=image, return_tensors="pt", padding=True)

    # Run through the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

    label = model.config.id2label[predicted_class_id]
    confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class_id].item()

    return {
        "result": label,
        "confidence": round(confidence * 100, 2)
    }

