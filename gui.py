import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class DigitDrawer:
    def __init__(self):
        self.model = load_model('DeepNeural_Handwriting_model.keras')
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
    def clear_canvas(self):
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
    def predict_digit(self):
        # Resize and invert the image
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted_image = ImageOps.invert(resized_image)
        
        # Convert image to a numpy array
        image_array = np.array(inverted_image)
        
        # Normalize the image array
        image_array = image_array / 255.0
        
        # Reshape the image for the model
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Predict the digit using the loaded model
        prediction = self.model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        
        return predicted_digit

def main():
    st.title("Digit Drawer")
    
    drawer = DigitDrawer()
    
    # Create a canvas component
    canvas_result = st.canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        drawer.image.paste(image, (0, 0))
    
    if st.button("Clear"):
        drawer.clear_canvas()
    
    if st.button("Predict"):
        predicted_digit = drawer.predict_digit()
        st.write(f"Predicted Digit: {predicted_digit}")
    
if __name__ == "__main__":
    main()
