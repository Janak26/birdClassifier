import tensorflow as tf
import os
import numpy as np
import cv2


trainPath = r"D:\Projects\BirdClassification\data\train"

reloaded = tf.saved_model.load(r'effNetCheckpoints')

def mapLabels():
	labelsMapper = {}
	for index,label in enumerate(os.listdir(trainPath)):
		labelsMapper[label] = index
	return labelsMapper


labeltoInt = mapLabels()
inttoLabel = {v:k for k,v in labeltoInt.items()}


def predict(imagePath):
	imageLoaded = cv2.imread(imagePath)
	imageLoaded = cv2.cvtColor(imageLoaded, cv2.COLOR_BGR2RGB)
	imageLoaded = tf.convert_to_tensor(imageLoaded, dtype=tf.float32)
	imageLoaded = tf.reshape(imageLoaded, shape=(1,224,224,3))
	prediction = reloaded.signatures['serving_default'](x=imageLoaded)
	birdName = inttoLabel[list(np.array(tf.argmax(prediction['output_0'], 1)))[0]]

	return birdName



if __name__ == "__main__":
	url = r"D:\Projects\BirdClassification\data\test\AMERICAN BITTERN\1.jpg"
	prediction = predict(url)
	print(prediction)