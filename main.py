from fastapi import FastAPI, UploadFile, File
from detector import classify_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ScrollSpy AI backend is live"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    result = classify_image(await file.read())
    return result
