from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

# ------------------ INIT APP ------------------
app = FastAPI(title="ANPR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODELS (ONCE) ------------------
model = YOLO("best.pt")
ocr = PaddleOCR(lang="en")
os.makedirs("temp", exist_ok=True)


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    # Read image from React (in-memory)
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"plate": None}

    img = cv2.resize(img, (640, 640))

    results = model(img)

    # Final values
    best_plate = ""

    # Text Extraction
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop license plate
            plate = img[y1:y2, x1:x2]

            # OCR
            ocr_result = ocr.predict(plate)

            if ocr_result and ocr_result[0]:
                texts = ocr_result[0]['rec_texts']
                scores = ocr_result[0]['rec_scores']
                for text, score in zip(texts, scores):
                    # Clean text
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                    # Filter likely plate strings
                    if score > 0.6 and len(clean_text) >= 8:
                        best_plate= clean_text

    # Post process text
    best_plate = best_plate.replace('O', '0').replace('I', '1')

    # Print the value
    return {"plate": best_plate or None}

@app.get("/")
def root():
    return {"status": "ANPR API running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
