from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt

CNN = tf.keras.models.load_model('C:\\Users\\nicol\\.spyder-py3\\CNN_perso.h5')

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
# pickle_in = open("data.pickle", "rb")
# data = pickle.load(pickle_in)




DIRECTORY = "C:\\Users\\nicol\\.spyder-py3\\Face Mask Dataset\\Test"


CATEGORIES = ['WithoutMask', 'WithMask', 'IncorrectlyWearedMask']
IMG_SIZE = 64 # IMG_SIZE = 224 alternative size

# #data
# X = []
# #labels(0,1,2)
# y = []

# def create_data():
#     for category in CATEGORIES:
#         path = os.path.join(DIRECTORY, category)
#         class_num_label = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
#                 img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
#                 X.append(img_array)
#                 y.append(class_num_label)
#             except Exception as e:
#                 pass
            
# create_data()

# # Get images as a 4,096 feature set
# SAMPLE_SIZE = len(y)
# data = np.array(X).flatten().reshape(SAMPLE_SIZE, IMG_SIZE*IMG_SIZE, 3) # pixel-features

# # Turn X and y into numpy arrays
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # images
# y = np.array(y) # target

# # print("Features, X shape: ", X.shape)
# # print("Target, y shape: ", y.shape)
# # print("Data shape: ", data.shape)


# # Saves us from having to regenerate our data by saving our data
# pickle_out = open("X1.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y1.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_out = open("data1.pickle", "wb")
# pickle.dump(data, pickle_out)
# pickle_out.close()

#-------------------------------------------------------------------------------------------#


pickle_in = open("X1.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y = pickle.load(pickle_in)
# pickle_in = open("data1.pickle", "rb")
# data = pickle.load(pickle_in)
# normalize the pixel values
X = X / 255



y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='int64')

y_pred = CNN.predict(X)
y_pred1 = np.argmax(y_pred,axis=1)
y1 = np.argmax(y,axis=1)

E = []
for k in range(len(y)) :
    E.append(y1[k]-y_pred1[k])

e = 0
for k in range(len(E)) :
    if E[k] != 0 : 
        e += 1
        
print("Prédiction :")
print("Nombre d'erreurs : ", e, "sur", len(y), ' ', "(" + str(round(e/len(E)*100, 3)) + "%" + ")")


    



# image = cv2.imread('C:\\Users\\nicol\\.spyder-py3\\IMG_4565.jpg')
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image = cv2.resize(image, (64,64))
# image = np.array(image).reshape(-1, 64, 64, 3)
# # image = image.reshape(-1,64,64,3)
# y_pred = np.argmax(CNN.predict(image))
# if y_pred == 0 :
#     print("L'individu ne porte pas de masque")
# if y_pred == 1 :
#     print("L'individu porte un masque")
