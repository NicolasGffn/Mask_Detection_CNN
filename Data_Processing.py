import numpy as np
import os
import cv2
import pickle


# On part d'un jeux de données organisés en dossiers d'images non labelisées
# Pour exploiter ce jeu de données, il faut associer un label à toutes les images de chaque sous-dossier


# On crée des données pour l'entraînement et pour le test du modèle :

DIRECTORY_TRAIN = "C:\\Users\\nicol\\.spyder-py3\\Face Mask Dataset\\Train"
DIRECTORY_TEST = "C:\\Users\\nicol\\.spyder-py3\\Face Mask Dataset\\Test"

CATEGORIES = ['WithoutMask', 'WithMask', 'IncorrectlyWornMask']
IMG_SIZE = 64 


# Data
X = []
# Labels(0,1,2)
y = []

def create_data():
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY_TRAIN, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X.append(img_array)
                y.append(class_num_label)
            except Exception as e:
                pass
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY_TEST, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X1.append(img_array)
                y1.append(class_num_label)
            except Exception as e:
                pass

create_data()


# Reshaping des données
SAMPLE_SIZE_TRAIN = len(y)
data = np.array(X).flatten().reshape(SAMPLE_SIZE, IMG_SIZE*IMG_SIZE, 3) # pixel-features

SAMPLE_SIZE_TEST = len(y1)
data1 = np.array(X1).flatten().reshape(SAMPLE_SIZE, IMG_SIZE*IMG_SIZE, 3) # pixel-features


# Transformation de X et y en array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
X = np.array(X1).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y1)

# Affichage des formats des jeux de données
print("Format de X : ", X.shape)
print("Format de y : ", y.shape)
print("Format des données : ", data.shape)


# On enregistre les données pour ne pas avoir à les générer à chaque fois
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("X1.pickle", "wb")
pickle.dump(X1, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_out = open("y1.pickle", "wb")
pickle.dump(y1, pickle_out)
pickle_out.close()

pickle_out = open("data.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()
pickle_out = open("data1.pickle", "wb")
pickle.dump(data1, pickle_out)
pickle_out.close()




