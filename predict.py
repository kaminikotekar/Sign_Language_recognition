# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:12:55 2018

@author: Kamini
"""

from keras.models import load_model
import cv2
import numpy as np

model= load_model('conv_model.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

img1=cv2.imread('testC1.jpg')
img2=cv2.imread('testB.jpg')
img1=cv2.resize(img1,(64,64))
img2=cv2.resize(img2,(64,64))
#img1=np.expand_dims(img1, axis=0)
#img2=np.expand_dims(img2, axis=0)

listl=[]
listl.append(img1)
listl.append(img2)
a=np.array(listl)
classes=model.predict_classes(a) 


