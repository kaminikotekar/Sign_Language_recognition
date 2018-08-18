# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:12:55 2018

@author: Kamini
"""

from keras.models import load_model
import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from Label import label_generator

def set_model():
    
    model= load_model('conv_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def prediction(model):
    
    os.chdir('.\\screens')
    img_list=[]
    for files in os.listdir('.\\'):
        #print(files)
        img=cv2.imread(files)
        img=cv2.resize(img,(64,64))
        img_list.append(img)
    
    os.chdir('..\\')
    
    a=np.array(img_list)
    predictions=model.predict_classes(a)
    #print(classes)
    labels= label_generator()
    final_predictions=predictions = [labels[k] for k in predictions]
    print(final_predictions)
    
    
def pred_image_only(model, img):
    
    pred_image=cv2.resize(img,(64,64))
    pred_image=np.expand_dims(pred_image, axis=0)
    predictions=model.predict_classes(pred_image)
    labels= label_generator()
    final_predictions=[labels[k] for k in predictions]
    print(final_predictions)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, final_predictions[0], (2,100), font, 2, (255, 0,0 ), 2, cv2.LINE_AA)


##img2=cv2.imread('testB.jpg')
#img1=cv2.resize(img1,(64,64))
#img2=cv2.resize(img2,(64,64))
#img1=np.expand_dims(img1, axis=0)
#img2=np.expand_dims(img2, axis=0)

#listl=[]
    
    
    
    

#listl.append(img1)
#listl.append(img2)
#a=np.array(listl)
#classes=model.predict_classes(a) 


#model.evaluate_generator(generator=test_set)
#test_set.reset()
#pred=model.predict_generator(test_set,verbose=1)
#predicted_class_indices=np.argmax(pred,axis=1)


