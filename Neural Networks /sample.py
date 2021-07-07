import tensorflow as tf
import pandas as pd
import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import activations
# 60k Images for training and 10k for testing
fashion = keras.datasets.fashion_mnist
(train_images ,train_lables ),(test_images ,test_lables )= fashion.load_data() # split training and testing
# print(train_images.shape)  #prints  10k 28 , 28 as a shape
#to see one pixel 
print(train_images[0,23,25]) #gives  0 
class_names = ["T-shirt/top", "Trouser" , "Pullover" ,"Dress", "Coat", 
"Sandal" , "Shirt" , "Sneaker", "Bag", "Ankle boot"]
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Data PreProcessing 
train_images = train_images / 255.0
test_images  = test_images / 255.0

model  =keras.Sequential([
    keras.layers.Flatten(input_shape = (28 ,28 ,)), 
    keras.layers.Dense(128 ,activations
])