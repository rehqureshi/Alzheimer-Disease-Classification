import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image

model=load_model('/Users/rehanqureshi/Desktop/Major_Project/code/alzheimers.h5')

img_path="/Users/rehanqureshi/Desktop/Major_Project/code/test/MildDemented/26 (19).jpg"
img_pred=cv2.imread("/Users/rehanqureshi/Desktop/Major_Project/code/test/MildDemented/26 (19).jpg")

def model_prediction(img_path,model):
    img=image.load_img(img_path,target_size=(208,176))
    predicted_data=np.expand_dims(img,axis=0)
    prediction=model.predict(predicted_data)

    i=0

    for i in range(4):
        if prediction[0][i]==1.00:
            if i==0:
                print('Mild Demented')
            elif i==1:
                print('Modearate Demented')
            elif i==2:
                print('No Alzheimer')
            else:
                print('Very Mild Demented')

        i+=1

model_prediction(img_path,model)

plt.imshow(img_pred)
plt.show()
