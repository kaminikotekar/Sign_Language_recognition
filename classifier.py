# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:10:35 2018

@author: Kamini
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# creating a sequential model
classifier= Sequential()

#adding layers
classifier.add(Convolution2D(32 ,(3 ,3) , activation="relu", input_shape=(64, 64, 3)))

classifier.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='valid' ))

#adding another hidden layer
classifier.add(Convolution2D(32, (3, 3), activation="relu"))

classifier.add(MaxPooling2D( pool_size=(2,2), strides=2, padding= 'valid'))

#Flattenin=\]45
classifier.add(Flatten())

#Full-connection step
#Dense(no of nodes in hiddemn layer, activation)
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(29, activation='softmax'))

#Compiling the cnn
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#fitting cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip= True)

test_datagen= ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64,64),#size for convolution part, resizing
                                                batch_size=300,
                                                class_mode='categorical'
                                                )

test_set= test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64,64),
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch =87000,
                         nb_epoch=5,
                         validation_data=test_set,
                         nb_val_samples=28)

classifier.save('conv_model.h5')
model.evaluate_generator(generator=test_set)