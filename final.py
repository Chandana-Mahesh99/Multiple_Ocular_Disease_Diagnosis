import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r'C:\Users\jayan\PycharmProjects\eye_project\eye_disease_classifier_finetuned.h5')

# Define the class names
class_names = ['Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal']


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1] range
    return img_array


def classify_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]

    print(f"Predicted class: {predicted_class_name}")
    print(f"Prediction confidence: {predictions[0][predicted_class[0]]:.2f}")

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class_name}")
    plt.axis('off')
    plt.show()


# Example usage
img_path = r"C:\Users\jayan\OneDrive\Desktop\eye_final\Testing\normal\image50.png"
classify_image(img_path)
