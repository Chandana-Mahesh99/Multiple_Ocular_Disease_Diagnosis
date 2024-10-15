import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Path to the directory containing the eye images
image_directory = r"C:\Users\jayan\OneDrive\Desktop\eye_all\datasets\hypertension"
output_directory = r"C:\Users\jayan\OneDrive\Desktop\eye_final\Training\hypertension"

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Create a data generator with specific augmentations
datagen = ImageDataGenerator(
    rotation_range=30,  # Random rotations
    width_shift_range=0.1,  # Random horizontal shifts
    height_shift_range=0.1,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest'  # Fill mode for new pixels
)

# Function to augment and save images
def augment_images(image_path, save_dir, prefix='augmented', num_augmented_images=5):
    img = load_img(image_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix, save_format='jpeg'):
        i += 1
        if i >= num_augmented_images:
            break  # We need only `num_augmented_images` images

# Loop through all images in the directory and augment them
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(image_directory, filename)
        augment_images(image_path, output_directory)

print("Image augmentation for pathological eye images completed.")
