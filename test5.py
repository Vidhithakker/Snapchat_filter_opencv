#Filter in over face

import cv2
from PIL import Image
import numpy as np
import time

maskPath = "filter1.png"

cascPath = "ff.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

mask = Image.open(maskPath)

def thug_mask(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(gray, 1.15)

	background = Image.fromarray(image)

	for (x,y,w,h) in faces:
		resized_mask = mask.resize((w,h), Image.ANTIALIAS)
		offset = (x,y)
		background.paste(resized_mask, offset, mask=resized_mask)

	return np.asarray(background)

cap = cv2.VideoCapture(0)

while True:
	flag, img = cap.read()

	cv2.imshow('VIDEO', thug_mask(img))

	cv2.waitKey(10)
