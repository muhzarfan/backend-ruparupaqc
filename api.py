from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fastapi.responses import JSONResponse

app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi
MODEL_PATH = 'model/furniture_model.h5'
IMAGE_SIZE = (128, 128)
CLASSES = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def load_furniture_model():
    """Load model h5"""
    try:
        model = load_model(MODEL_PATH)
        print("Model berhasil dimuat")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_image(img):
    """Prepare image for prediction"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load model
model = load_furniture_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Process image
        processed_img = prepare_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img)
        
        # Get results
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Prepare response
        result = {
            'status': 'success',
            'class': predicted_class,
            'confidence': f'{confidence * 100:.2f}%',
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(CLASSES, predictions[0])
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'error': str(e),
                'type': str(type(e).__name__)
            }
        )

@app.get("/health")
async def health():
    if model is None:
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': 'Model not loaded'}
        )
    return {'status': 'healthy', 'message': 'Service is running'}