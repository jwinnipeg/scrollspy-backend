from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from detector import classify_image
from PIL import Image
import io

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classify_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "ScrollSpy AI is live"}