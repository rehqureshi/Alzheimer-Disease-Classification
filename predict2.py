import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained CNN model
model = load_model('/Users/rehanqureshi/Desktop/Major_Project/code/snapshot_1.hdf5')

# Image path
img_path = "/Users/rehanqureshi/Desktop/Major_Project/code/test/MildDemented/26 (19).jpg"

# Load image using OpenCV for visualization
img_pred = cv2.imread(img_path)
img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

def model_prediction(img_path, model):
    # Load and preprocess image for the model
    img = image.load_img(img_path, target_size=(176, 176), color_mode='grayscale')
    img_array = image.img_to_array(img)         # shape: (176, 176, 1)
    img_array = img_array / 255.0                # normalize to 0-1
    predicted_data = np.expand_dims(img_array, axis=0)  # shape: (1, 176, 176, 1)

    print("Input shape to model:", predicted_data.shape)

    # Perform prediction
    prediction = model.predict(predicted_data)
    class_index = np.argmax(prediction[0])

    classes = ['Mild Demented', 'Moderate Demented', 'No Alzheimer', 'Very Mild Demented']
    print(f"Prediction: {classes[class_index]}")

    return img_pred

# Run the prediction
img_to_show = model_prediction(img_path, model)

# Display the image
plt.imshow(img_to_show)
plt.axis('off')
plt.title("Predicted Image")
plt.show()
