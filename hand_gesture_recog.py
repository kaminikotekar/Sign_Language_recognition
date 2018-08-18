# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:08:00 2018

@author: Kamini
"""

import cv2
import numpy as np
from predict import set_model, prediction
from predict import pred_image_only

cap=cv2.VideoCapture(1)
fgbg= cv2.createBackgroundSubtractorMOG2()
model=set_model()
i=1
while True:
    ret, img= cap.read()
    #gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #blurring the image
    
    blur=cv2.GaussianBlur(img,(15,15),0)
    
    #Applying threshold
    #th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
    
    fgmask=fgbg.apply(blur)
    Colored_Mask = cv2.bitwise_and(img, img, mask=fgmask)
    nzCount = cv2.countNonZero(fgmask)
    if nzCount >= 5000 and nzCount<307200:
        cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=img)
        pred_image_only(model, img)
        i+=1
    
    cv2.imshow('image', img)
    cv2.imshow('thresh', Colored_Mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
prediction(model)  