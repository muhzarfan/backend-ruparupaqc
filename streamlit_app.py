# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io

# Konfigurasi
API_URL = "https://your-fastapi-app.onrender.com"  # Ganti dengan URL FastAPI Anda

def predict_image(image_file):
    """Send image to FastAPI endpoint for prediction"""
    files = {'file': ('image.jpg', image_file, 'image/jpeg')}
    try:
        response = requests.post(f"{API_URL}/predict", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title('Furniture Classification App')
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Display image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Predict'):
                with st.spinner('Processing...'):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get prediction from API
                    result = predict_image(uploaded_file)
                    
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"Predicted Class: {result['class']}")
                        st.info(f"Confidence: {result['confidence']}")
                        
                        st.subheader('Class Probabilities:')
                        probabilities = result['probabilities']
                        for class_name, prob in probabilities.items():
                            st.write(f'{class_name}: {prob * 100:.2f}%')
                            st.progress(float(prob))
                        
        except Exception as e:
            st.error(f'Error processing image: {str(e)}')

if __name__ == '__main__':
    main()