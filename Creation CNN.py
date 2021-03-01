from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
# # import PIL
import cv2
import pickle
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt

# Création d'une architecture classique d'un CNN
# 2 couches de convolution (avec fonction d'activation ReLU pour chacune d'elles) 
# Une couche de pooling
# 2 couches fully connected suivies d'une fonction ReLU pour chacune d'elles
# Une couche fully connected de classification

def creation_CNN() :

    CNN = Sequential()

    # Ajout de la première couche de convolution, suivie d'une couche ReLU
    CNN.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu'))

    # Ajout de la deuxième couche de convolution, suivie  d'une couche ReLU
    CNN.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    # Ajout de la première couche de pooling
    CNN.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
    
    CNN.add(Flatten())  # Conversion des matrices 3D en vecteur 1D

    # Ajout de la première couche fully-connected, suivie d'une couche ReLU
    CNN.add(Dense(4096, activation='relu'))
    
    # Ajout de la deuxième couche fully-connected, suivie d'une couche ReLU
    CNN.add(Dense(4096, activation='relu'))
    
    # Ajout de la dernière couche fully-connected qui permet de classifier
    CNN.add(Dense(1000, activation='softmax'))


# Le modèle créé est trop coûteux en mémoire pour être implémenté sur une machine classique
# De plus, il faut maintenant l'entraîner, ce qui peut prendre 1 mois
# Il est donc préférable d'utiliser un modèle pré-construit et pré-entraîné, en réalisant du transfer-learning









#-----------------------------------------------------------------------------------------------------------------#







# On récupère un CNN déjà entraîné afin de réaliser du fine-tuning total.
# Pour se faire, on enlève les couches fully connected du réseau (qui servent à classifier selon des catégories de base)
# et on crée les couches qui seront utilisées pour notre classification



# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
 
for layer in model.layers:
   layer.trainable = False


# Récupérer la sortie de ce réseau
output_vgg16_conv = model.output


# Ajouter la nouvelle couche fully-connected pour la classification à 2 classes

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(3, activation='softmax')(x)


# Définir le nouveau modèle
CNN = keras.models.Model(inputs=model.input, outputs=x)






# Compiler le modèle 
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])








DIRECTORY = "C:\\Users\\nicol\\.spyder-py3\\Face Mask Dataset\\Train"

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
# pickle_out = open("X.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_out = open("data.pickle", "wb")
# pickle.dump(data, pickle_out)
# pickle_out.close()



pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
# pickle_in = open("data.pickle", "rb")
# data = pickle.load(pickle_in)

print('# of Samples:', len(y))
print('# of Without A Mask:', (y == 0).sum())
print('# of With A Mask:', (y == 1).sum())
print('# of With An Incorrectly Weared Mask:', (y == 2).sum())


print('Shape of X:', X.shape)
print('Shape of y:', y.shape)




# # resized_X = []
# # for img in X:
# #     resized_X.append(cv2.resize(img, (64, 64)))

# # X = np.asarray(resized_X)
# X = X.reshape(-1, 64, 64, 3)
# print(X.shape)


# normalize the pixel values
X = X / 255

# plt.imshow(X[5014])
# plt.show()
   
# IMG_DIM = X.shape[1]
# print('IMG_DIM:',IMG_DIM)




# # Split our data into testing and training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=45)

# y_train = tf.keras.utils.to_categorical(y_train, num_classes=None, dtype='float32')
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=None, dtype='float32')

y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')


# Print the length and width of our testing data.
# print('Length of our Training data: ', len(X_train), '\nLength of our Testing data: ', len(X_test))


# Entraîner sur les données d'entraînement (X_train, y_train)
history = CNN.fit(X, y, epochs=20, verbose=1)

# Summarize history for train
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.show()





CNN.save('C:\\Users\\nicol\\.spyder-py3\\CNN_perso.h5')

