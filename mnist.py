#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:23:29 2021

@author: Gabriel C.
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[5], cmap='gray')
plt.title('Class ' + str(y_train[5]))

previews_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

previews_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

previews_train = previews_train.astype('float32')
previews_test = previews_test.astype('float32')

previews_train /= 255.0
previews_test /= 255.0

class_train = np_utils.to_categorical(y_train, 10)

class_test = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(28, 28, 1),
                      activation='relu',))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2, 2)))

#classifier.add(Flatten())

classifier.add(Conv2D(32, (3, 3), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'],)

classifier.fit(previews_train, class_train, batch_size=128,
               epochs=5, validation_data=(previews_test, class_test))
