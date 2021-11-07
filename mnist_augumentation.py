#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:57:21 2021

@author: Gabriel C.
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = mnist.load_data()

previews_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
previews_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
previews_train = previews_train.astype('float32')
previews_test = previews_test.astype('float32')
previews_train /= 255.0
previews_test /= 255.0
class_train = np_utils.to_categorical(y_train, 10)
class_test = np_utils.to_categorical(y_test, 10)


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), 
               activation='relu'))

classifier.add(MaxPooling2D())
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])



generator_train = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                     shear_range=0.2, height_shift_range=0.07,
                                     zoom_range=0.2)


generator_test = ImageDataGenerator()

train_database = generator_train.flow(previews_train, class_train,
                                      batch_size=128)

test_database = generator_test.flow(previews_test, class_test,
                                    batch_size=128)

classifier.fit_generator(train_database, steps_per_epoch=60000/128,
                         epochs=5, validation_data= test_database,
                         validation_steps=10000/128)