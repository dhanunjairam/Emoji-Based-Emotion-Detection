import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
import os 
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import threading
from tensorflow.keras.models import load_model #type:ignore


# Load the model
model = load_model('best_model.keras')
model.load_weights('model2.weights.h5')


class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emoji_paths = {
    'angry': ' ',#add angry image path
    'disgust': '',#add disgust image path
    'fear': ' ',#add fear image path
    'happy': '',#add happy image path
    'neutral': '',#add neutral image path
    'sad': '',#add sad image path
    'surprise': ''#add surprise image path
}

def preprocess_image(image_path):
    datagen = ImageDataGenerator(rescale=1./255)  # Normalizing the image
    target_size=(48, 48)
    color_mode='grayscale'
    # Load the image using OpenCV
    if color_mode == 'grayscale':
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)  # For RGB
    
    if img is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    
    # Resize the image to the required size
    img = cv2.resize(img, target_size)
    
    # Reshape the image to (batch_size=1, target_size[0], target_size[1], channels)
    if color_mode == 'grayscale':
        img = np.expand_dims(img, axis=-1)  # Add the channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 48, 48, 1) for grayscale or (1, 48, 48, 3) for RGB
    
    # Use the ImageDataGenerator to process the image
    img_gen = datagen.flow(img, batch_size=1)
    
    return img_gen

def predict_expression(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Predict the class
    predictions = model.predict(img)
    
    # Get the predicted class
    class_index = np.argmax(predictions)
    
    # Return the predicted expression
    return class_labels[class_index]



def show_expression_and_emoji(image_path):
    # Predict the expression
    expression = predict_expression(image_path)
    
    # Update the label with the predicted expression
    expression_label.config(text=f'Predicted Expression: {expression}')
    
    # Load and display the corresponding emoji
    emoji_img_path = emoji_paths[expression]
    emoji_img = Image.open(emoji_img_path)
    emoji_img = emoji_img.resize((100, 100), Image.Resampling.LANCZOS)  # Resize the emoji
  # Resize the emoji
    emoji_img_tk = ImageTk.PhotoImage(emoji_img)
    
    # Update the emoji label
    emoji_label.config(image=emoji_img_tk)
    emoji_label.image = emoji_img_tk  # Keep a reference to avoid garbage collection

# Tkinter setup
root = tk.Tk()
root.title("Facial Expression and Emoji Display")

# Expression label
expression_label = Label(root, text="", font=("Helvetica", 16))
expression_label.pack(pady=20)

# Emoji label (where emoji will be displayed)
emoji_label = Label(root)
emoji_label.pack(pady=20)

# Button to show the expression and emoji
image_path = '/Users/ABC/EMOJIFY /hapy.png'  # Replace with your image path
button = Button(root, text="Show Expression and Emoji", command=lambda: show_expression_and_emoji(image_path))
button.pack(pady=20)

root.mainloop()


#surprise.png
#sad.png
#neutral.png
#angry.jpg
#fear.png
#hapy.png