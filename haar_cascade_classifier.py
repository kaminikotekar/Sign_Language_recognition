# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:10:28 2018

@author: Kamini
"""

import numpy as np
import cv2
from predict import set_model, prediction


open_palm=cv2.CascadeClassifier('open_palm.xml')
closed_palm=cv2.CascadeClassifier('closed_palm.xml')
hand= cv2.CascadeClassifier('hand.xml')
agest= cv2.CascadeClassifier('aGest.xml')



cap= cv2.VideoCapture(1)

i=1
while True:
    _, img= cap.read()
    #gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frameId = cap.get(1)
    op= open_palm.detectMultiScale(img,1.3,5)
    cp= closed_palm.detectMultiScale(img,1.3,5)
    h = hand.detectMultiScale(img,1.3,5)
    a=  agest.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in op:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=img)
        i+=1
    for (x,y,w,h) in cp:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=img)
        i+=1
        
    for (x,y,w,h) in a:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=img)
        i+=1
    
        
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
model=set_model()
prediction(model)
