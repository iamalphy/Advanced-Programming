import numpy as np
import pandas as pd
import tensorflow as tf
import os


# Image related
import cv2
from PIL import Image

# For ploting
import matplotlib.pyplot as plt

# For the model and it's training
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

os.sys.path

mymodel=load_model('traffic_classifier.h5')
image = Image.open('./'+img)
image = image.resize([30, 30])
data.append(np.array(image))
X_test=np.array(data)
pred = np.argmax(mymodel.predict(X_test), axis=-1)
print(pred)