from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
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

# Création d'une architecture classique d'un CNN
# 2 couches de convolution (avec fonction d'activation ReLU pour chacune d'elles) 
# Une couche de pooling
# 2 couches fully connected suivies d'une fonction ReLU pour chacune d'elles
# Une couche fully connected de classification


# Il est donc préférable d'utiliser un modèle pré-construit et pré-entraîné, en réalisant du transfer-learning







# On récupère un CNN déjà entraîné afin de réaliser du fine-tuning total.
# Pour se faire, on enlève les couches fully connected du réseau (qui servent à classifier selon des catégories de base)
# On crée ensuite les couches qui seront utilisées pour notre classification



# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
 
for layer in model.layers:
   layer.trainable = False


# Récupérer la sortie de ce réseau
output_vgg16_conv = model.output


# Ajouter les npouvelles couches fully-connected pour la classification à 3 classes
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
x = Dense(3, activation='softmax')(x)


# Définir le nouveau modèle
CNN = keras.models.Model(inputs=model.input, outputs=x)

# Compiler le modèle 
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])






# On charge les données générées
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


# Affichage des caractéristiques des jeux de données
print('# of Samples:', len(y))
print('# of Without A Mask:', (y == 0).sum())
print('# of With A Mask:', (y == 1).sum())
print('# of With An Incorrectly Worn Mask:', (y == 2).sum())


# Affichage test d'une image
plt.imshow(X[5014])
plt.show()
   
# On normalise les valeurs des pixels
X = X / 255

# Transformation de y nécessaire pour l'entraînement
y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')

# Entraînement sur les données X et y
history = CNN.fit(X, y, epochs=20, verbose=1)


# Affichage des courbes de précision et de perte
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Précision et perte du modèle')
plt.ylabel('Précision/Perte')
plt.xlabel('epoch')
plt.legend(['Précision', 'Perte'], loc='right')
plt.show()




# Sauvegarde du modèle entraîné
CNN.save(Lien du fichier)
