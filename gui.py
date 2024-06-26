import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Drawer")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.button_save = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.button_save.pack()
        
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load the Keras model
        self.model = load_model('DeepNeural_Handwriting_model.keras')
        
    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
    def predict_digit(self):
        # Resize and invert the image
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted_image = ImageOps.invert(resized_image)
        
        # Convert image to a numpy array
        image_array = np.array(inverted_image)
        
        # Normalize the image array
        image_array = image_array
        
        # Reshape the image for the model
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Predict the digit using the loaded model
        prediction = self.model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        
        # Display the prediction
        print(f"Predicted Digit: {predicted_digit}")
        self.show_prediction(predicted_digit)
        
    def show_prediction(self, digit):
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction")
        
        label = tk.Label(result_window, text=f"Predicted Digit: {digit}", font=('Helvetica', 24))
        label.pack()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
