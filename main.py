import os
import easyocr
import cv2
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import numpy as np
import uvicorn
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the fake news classifier (DistilBART)
fake_news_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

# Initialize FastAPI app
app = FastAPI()

# OCR function
def perform_ocr(image: Image.Image):
    reader = easyocr.Reader(['en'])
    image = np.array(image)
    results = reader.readtext(image)
    
    extracted_text = " ".join([text for (_, text, _) in results])
    return extracted_text.strip()

# Function to classify text using DistilBART
def classify_with_distilbart(text: str):
    candidate_labels = ["true", "fake"]
    classification = fake_news_classifier(text, candidate_labels)
    return classification["labels"][0]  # "true" or "fake"

# Function to classify text using Gemini API
def classify_with_gemini(text: str):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Classify the following news text as 'true' or 'fake': {text}")
    
    if response and response.text:
        gemini_result = response.text.lower()
        if "true" in gemini_result:
            return "true"
        elif "fake" in gemini_result:
            return "fake"
    return "unknown"

# Function to determine final classification
def determine_final_result(text: str):
    bart_label = classify_with_distilbart(text)
    gemini_label = classify_with_gemini(text)
    
    return bart_label if bart_label == gemini_label else gemini_label

# API endpoint to process images
app = FastAPI()

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(None)):  # Allow file to be optional
    if not file:
        return {"error": "No file uploaded!"}
    
    return {"filename": file.filename, "content_type": file.content_type}

# Run the API (for local testing)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
