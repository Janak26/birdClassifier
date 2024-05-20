import requests
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from datetime import datetime
import random
import os
import logging



logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='example.log', level=logging.DEBUG)

downloadsPath = r"D:\Projects\BirdClassification\downloads"


def createFilename():
	microsTime = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
	rnumber = str(random.randint(0, 9))
	fileName = microsTime + rnumber + '.jpg'
	fileName = os.path.join(downloadsPath, fileName)
	return fileName


def download_image(image_url):
	image = io.imread(image_url)
	image = cv2.resize(image, (224,224))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	savePath = createFilename()
	cv2.imwrite(savePath, image)
	logging.info('downloaded_image {} {}'.format(image_url, savePath)) 
	return savePath




if __name__ == "__main__":
	download_image("https://m.media-amazon.com/images/I/71hrBTIgLCL._AC_UF1000,1000_QL80_.jpg")