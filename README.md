# Emoji-Based-Emotion-Detection

Project Summary

This project leverages deep learning to detect facial expressions and display corresponding emojis. By analyzing an input image, it predicts the emotion and maps it to a visual emoji representation. The app is built using TensorFlow/Keras and Tkinter, with a trained CNN model for emotion classification. It supports multiple expressions, including happiness, sadness, anger, and more, making it an engaging and interactive experience.
Dependencies

Ensure you have the following dependencies installed:

    Python 3.10+
    TensorFlow/Keras
    OpenCV
    NumPy
    Pillow
    Tkinter

Install all required packages using the command:

pip install -r requirements.txt

Usage Instructions

    Clone the Repository:
    Download the project files to your local system.

    Set Up the Environment:
        Place the model files (model2.keras and model2.weights.h5) in the project root directory.
        Verify that emoji images are correctly placed in the /emojis/ folder and paths are correctly defined in the code.

    Update the Input Image Path:
        Open emoji.py in your preferred text editor.
        Locate line 101 and update the image_path variable to point to the image you want to analyze.
        Example:

    image_path = '/path/to/your/image.jpg'

Run the Application:
Start the app by running:

    python emoji.py

    View Results:
        The app processes the image and predicts the facial emotion.
        It displays the emotion alongside the corresponding emoji in a Tkinter window.

    Test with Another Image:
        Close the app.
        Update the image_path with the path to a new image and re-run the application.

Features

    Emotion Detection: Classifies expressions like happy, sad, angry, neutral, and more.
    Emoji Representation: Maps detected emotions to expressive emojis for visualization.
    Interactive Interface: Displays results in a user-friendly Tkinter window.
    Scalability: Potential for expansion to support hand gestures and additional expressions.

Future Enhancements

    Support for real-time emotion detection using a webcam.
    Integration with hand gesture recognition.
    Extending the emotion set for more granular detection.

Feel free to share your feedback or suggestions to improve the app! 😊
