import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model=load_model('alzheimers.h5')

test=ImageDataGenerator(1./255)

test_set=test.flow_from_directory('test/',target_size=(208,176),batch_size=3,class_mode='categorical')

test_acc=model.evaluate(test_set)

print('Test accuracy:',test_acc)

model.summary()