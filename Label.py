# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 18:12:17 2018

@author: Kamini
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip= True)


training_set=train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64,64),#size for convolution part, resizing
                                                batch_size=300,
                                                class_mode='categorical'
                                                )



def label_generator():
    
    labels = (training_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    return labels