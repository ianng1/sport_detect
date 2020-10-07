#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:42:59 2020

@author: pengwah
"""

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

import os 
execution_path = '/Users/pengwah/Desktop/Sport Detection Project'




image_classifier = Sequential()
image_classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
image_classifier.add(MaxPooling2D(pool_size = (8, 8)))
image_classifier.add(Flatten())


image_classifier.add(Dense(activation = 'relu', units = 8))
image_classifier.add(Dense(activation = 'relu', units = 8))
image_classifier.add(Dense(activation = 'relu', units = 8))
image_classifier.add(Dense(activation = 'relu', units = 8))
image_classifier.add(Dense(activation = 'softmax', units = 3))
image_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)


for file in os.listdir(execution_path + '/train'):
    print(file)

training_set = train_datagen.flow_from_directory(
        execution_path + '/train',
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        execution_path + '/test',
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')

image_classifier.fit_generator(
        training_set,
        steps_per_epoch=161,
        epochs=25,
        validation_data=test_set,
        validation_steps=48)
image_classifier.save(execution_path + "/image_classifier.h5")



testing = load_model(execution_path + "/image_classifier.h5")
testing.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



results = []
tennis_val = []
soccer_val = []
football_val = []

for filename in os.listdir(execution_path + "/validation/tennis"):
    if "person" in filename:
        tennis_val.extend(filename)
for filename in os.listdir(execution_path + "/validation/soccer"):
    if "person" in filename:
sfor filename in os.listdir(execution_path + "/validation/soccer"):
    if "person" in filename:
        soccer_val.extend(filename)        


#[football, soccer, tennis]
def test(sport):
    total = [0,0,0]
    target = execution_path + "/validation/" + sport
    for filename in os.listdir(target):
        if filename.endswith(".png"):
            name = execution_path + "/validation/" + sport + "/" + filename
            print(name)
            img = image.load_img(name, target_size=(128, 128))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            z = testing.predict(y)
            pred = testing.predict_classes(y)
            total[pred[0]] += 1
            print(pred)
            print(z)
    return total


def test_folder(path):

    total = [0,0,0]
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = image.load_img(path, target_size=(128, 128))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            z = testing.predict(y)
            pred = testing.predict_classes(y)
            total[pred[0]] += 1
            print(pred)
            print(z)
    return total




print(test('tennis'))
print(test('basketball'))    
print(test('football'))




