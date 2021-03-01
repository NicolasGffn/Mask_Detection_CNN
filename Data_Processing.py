import numpy as np
import os
import cv2
import pickle


# On part d'un jeux de données organisés en dossiers d'images non labelisées
# Pour exploiter ce jeu de données, il faut associer un label à toutes les images de chaque sous-dossier


DIRECTORY = "C:\\Users\\nicol\\.spyder-py3\\Face Mask Dataset\\Train"

CATEGORIES = ['WithoutMask', 'WithMask', 'IncorrectlyWornMask']
IMG_SIZE = 64 



# Data
X = []
# Labels(0,1,2)
y = []

def create_data():
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X.append(img_array)
                y.append(class_num_label)
            except Exception as e:
                pass

create_data()


# Reshaping des données
SAMPLE_SIZE = len(y)
data = np.array(X).flatten().reshape(SAMPLE_SIZE, IMG_SIZE*IMG_SIZE, 3) # pixel-features

# Transformation de X et y en array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # images
y = np.array(y) # target

# Affichage de la taille des jeux de données
print("Features, X shape: ", X.shape)
print("Target, y shape: ", y.shape)
print("Data shape: ", data.shape)




# On enregistre les données pour ne pas avoir à les générer à chaque fois
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("data.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()