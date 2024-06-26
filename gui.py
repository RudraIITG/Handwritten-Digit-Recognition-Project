import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

class DigitDrawer:
    def __init__(self):
        # Load the model with compile=False
        self.model = tf.keras.models.load_model('DeepNeural_Handwriting_model.keras', compile=False)
        # Initialize image and drawing
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        # Preprocess the image for prediction
        img = self.image.resize((28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0  # Normalize the pixel values

        # Predict digit using the loaded model
        prediction = self.model.predict(img_array)
        return prediction.argmax()

def main():
    st.title("Digit Drawer")
    drawer = DigitDrawer()

    # Create a canvas component
    canvas_result = st.canvas(width=280, height=280, drawing_mode="freedraw", key="canvas")

    # Recognize button to predict the digit
    if st.button("Recognize"):
        drawer.image.paste(canvas_result.image_data, (0, 0))
        predicted_digit = drawer.predict_digit()
        st.write(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
