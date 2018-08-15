# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:10:28 2018

@author: Kamini
"""

import numpy as np
import cv2


hand_cascade=cv2.CascadeClassifier('Hand.Cascade.1.xml')

cap= cv2.VideoCapture(1)

while True:
    _, img= cap.read()
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hand= hand_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in hand:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()