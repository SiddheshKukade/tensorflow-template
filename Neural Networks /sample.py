import tensorflow as tf
import pandas as pd
import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt

# 60k Images for training and 10k for testing
fashion = keras.datasets.fashion_mnist
(train_images ,train_lables ),(test_images ,test_lables )= fashion.load_data() # split training and testing
# print(train_images.shape)  #prints  10k 28 , 28 as a shape
#to see one pixel 
print(train_images[0,23,25]) #gives  0 
print(type(train_lables[:10))

class_names = ["T-shirt/top", "Trouser" , "Pullover" ,"Dress", "Coat", 
"Sandal" , "Shirt" , "Sneaker", "Bag", "Ankle boot"]

plt.figure()
plt.in