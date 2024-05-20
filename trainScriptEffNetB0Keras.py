import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
from scipy.ndimage import shift


import time
import math

print('Libraries imported')



trainPath = "/home/Projects/BirdClassification/data/train"
testPath = "/home/Projects/BirdClassification/data/test"
valPath = "/home/Projects/BirdClassification/data/valid"



batch_size = 128
num_classes = 525 # number of classes to train on, train on lower classes to test the code
endnumImages = 200 # number of images per class to train on, train on lower number of images to test the code 



# convert bird names to integers
def mapLabels():
	labelsMapper = {}
	birdFoldersMap = sorted(os.listdir(trainPath))
	for index,label in enumerate(birdFoldersMap):
		labelsMapper[label] = index
	return labelsMapper


labeltoInt = mapLabels()
inttoLabel = {v:k for k,v in labeltoInt.items()}




# image augmentation, rotating the image, code obtained from stackoverflow
def rotate_image(mat, angle):
	"""
	Rotates an image (angle in degrees) and expands image to avoid cropping
	"""
	
	height, width = mat.shape[:2] # image shape has 3 dimensions
	image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
	
	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
	
	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])
	
	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	
	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]
	
	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	rotated_mat = cv2.resize(rotated_mat, (224,224))
	return rotated_mat



# image augmentation, flip an image, code obtained from stackoverflow
def flip_image(mat):
	flipped = cv2.flip(mat, 1)
	return flipped



# image augmentation, shift an image horizontally, vertically or both ways
def imageShifter(image, shiftMode):
	if shiftMode == 'vertical':
		moveNum = random.randint(-15, 15)
		shiftedIm = shift(image, [moveNum, 0, 0], mode='constant')
		return shiftedIm
	if shiftMode == 'horizontal':
		moveNum = random.randint(-15, 15)
		shiftedIm = shift(image, [0, moveNum, 0], mode='constant')
		return shiftedIm
	if shiftMode == 'both':
		moveH = random.randint(-15, 15)
		moveV = random.randint(-15, 15)
		shiftedIm = shift(image, [moveV, moveH, 0], mode='constant')
		return shiftedIm



# create a list of filenames and randomize them for training
def createFilenames(mode, filenameBatchSize=None):
	filenames = []
	labelnames = []
	if mode == 'train':
		dataPath = trainPath
	elif mode == 'val':
		dataPath = valPath

	birdsFolders = sorted(os.listdir(dataPath))
	birdsFolders = birdsFolders[:num_classes]
	for bird in birdsFolders:
		imagesPath = os.listdir(os.path.join(dataPath, bird))
		birdImages = sorted(os.listdir(os.path.join(dataPath, bird)))
		if mode == 'train':
			birdImages = birdImages[:endnumImages]
		for singleImage in birdImages:
			filenames.append(os.path.join(dataPath, bird, singleImage))
			labelnames.append(labeltoInt[bird])


	random.Random(4).shuffle(filenames)
	random.Random(4).shuffle(labelnames)

	return filenames, labelnames



# image generator to save memory while training, parts of code obtained from ChatGPT
class CustomDataGenerator:
	def __init__(self, image_paths, labels, image_size=(224, 224), batch_size=32, shuffle=True):
		self.image_paths = image_paths
		self.labels = labels
		self.image_size = image_size
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(image_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def generate_data(self):
		num_batches = len(self.image_paths) // self.batch_size
		while True:
			for i in range(num_batches):
				images_batch = []
				labels_batch = []
				start_idx = i * self.batch_size
				end_idx = (i + 1) * self.batch_size
				batch_indexes = self.indexes[start_idx:end_idx]
				for idx in batch_indexes:
					image_path = self.image_paths[idx]
					image = cv2.imread(image_path)
					image = cv2.resize(image, self.image_size)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					label = self.labels[idx]

					random_processor = random.randint(0,29)

					rotate_degree = random.randint(10,25)

					if random_processor < 6:
						images_batch.append(image)
						labels_batch.append(label)

					elif 5 < random_processor < 11:	
						processed_image = rotate_image(image, 20)
						images_batch.append(processed_image)
						labels_batch.append(label)

					elif 10 < random_processor < 16:
						processed_image = flip_image(image)
						images_batch.append(processed_image)
						labels_batch.append(label)

					elif 15 < random_processor < 21:
						processed_image = rotate_image(image, rotate_degree)
						processed_image = flip_image(processed_image)
						images_batch.append(processed_image)
						labels_batch.append(label)

					elif 20 < random_processor < 25:
						processed_image = imageShifter(image, 'horizontal')
						images_batch.append(processed_image)
						labels_batch.append(label)


					elif 24 < random_processor < 28:
						processed_image = imageShifter(image, 'vertical')
						images_batch.append(processed_image)
						labels_batch.append(label)


					elif 27 < random_processor < 30:
						processed_image = imageShifter(image, 'both')
						images_batch.append(processed_image)
						labels_batch.append(label)

				yield np.array(images_batch), np.array(labels_batch)




# Define the EfficientNet model
class CNNModel(tf.keras.Model):
	def __init__(self, num_classes):
		super(CNNModel, self).__init__()
		self.efficientnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
		self.pool = GlobalAveragePooling2D()
		self.fc1 = Dense(256, activation='relu')
		self.bn1 = BatchNormalization()
		self.dropout1 = Dropout(rate=0.5)
		self.fc2 = Dense(256, activation='relu')
		self.bn2 = BatchNormalization()
		self.dropout2 = Dropout(rate=0.5)
		self.fc3 = Dense(num_classes, activation='softmax')

	def call(self, inputs, training=False):
		x = self.efficientnet(inputs, training=training)
		x = self.pool(x)
		x = self.fc1(x)
		x = self.bn1(x, training=training)  # Use training argument here
		x = self.dropout1(x, training=training)
		x = self.fc2(x)
		x = self.bn2(x, training=training)  # Use training argument here
		x = self.dropout2(x, training=training)
		return self.fc3(x)





model = CNNModel(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



print('Loading train data')
trainData, trainLabels = createFilenames('train')

train_generator = CustomDataGenerator(trainData, trainLabels, image_size=(224, 224), batch_size=batch_size, shuffle=False)

print('Loading val data')
valData, valLabels = createFilenames('val')

val_generator = CustomDataGenerator(valData, valLabels, image_size=(224, 224), batch_size=batch_size, shuffle=False)


print('Loaded all data')


# 20 epochs should be suffucuent to get high enough accuracy
num_epochs = 20


history = model.fit(train_generator.generate_data(),
					epochs=num_epochs,
					steps_per_epoch=len(trainData) // train_generator.batch_size,
					validation_data=val_generator.generate_data(),
					validation_steps=len(valData) // val_generator.batch_size)




@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def serve(x):
	return model(x)


tf.saved_model.save(model, '/home/Projects/BirdClassification/effNetCheckpoints/', signatures={'serving_default': serve})

hist_df = pd.DataFrame(history.history)

hist_df.to_csv('/home/Projects/BirdClassification/effNetAcc.csv', index=False)