import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile
import os
import shutil
from sklearn.model_selection import train_test_split


def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))

	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]
		# randomly pick an image that belongs to the *same* class
		# label

		idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
	
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
					   keepdims=True)
	
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
	
def import_data(data_path):
    zip_path = os.path.join(data_path, "data.zip")

    with ZipFile(zip_path, 'r') as zip:
        zip.extractall(data_path)
        
    base_path = os.path.join(data_path, "data/")
    output_path = os.path.join(data_path, "stanford-dogs-dataset-split/")
    train_folder = os.path.join(output_path, "train")
    test_folder = os.path.join(output_path, "test")

    # Criando as pastas de saída
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterar sobre os subdiretórios de raças
    for breed in os.listdir(base_path):
        breed_path = os.path.join(base_path, breed)
        if os.path.isdir(breed_path):  # Verifica se é uma pasta
            images = [img for img in os.listdir(breed_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Divide as imagens em treino e teste (80%-20%)
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

            # Criando as pastas específicas para cada raça
            train_breed_folder = os.path.join(train_folder, breed[10:])
            test_breed_folder = os.path.join(test_folder, breed[10:])
            os.makedirs(train_breed_folder, exist_ok=True)
            os.makedirs(test_breed_folder, exist_ok=True)

            # Movendo os arquivos para treino
            for img in train_images:
                shutil.copy(os.path.join(breed_path, img), os.path.join(train_breed_folder, img))

            # Movendo os arquivos para teste
            for img in test_images:
                shutil.copy(os.path.join(breed_path, img), os.path.join(test_breed_folder, img))