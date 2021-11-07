#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:53:26 2021

@author: Gabriel C.
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5

np.random.seed(seed)

(x, y), (x_test, y_test) = mnist.load_data()

previews = x.reshape(x.shape[0], 28, 28, 1)

previews = previews.astype('float32')

previews /= 255.0

class_ = np_utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = []

a = np.zeros(seed)

b = np.zeros(shape=(class_.shape[0], 1))

for index_train, index_test in kfold.split(previews,
                                           np.zeros(shape=(class_.shape[0], 1))):
    
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),
                   activation='relu'))
    classifier.add(MaxPooling2D())
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy'])
    
    classifier.fit(previews[index_train], class_[index_train],
                   batch_size=128, epochs=5)
    
    precision = classifier.evaluate(previews[index_train], class_[index_train])
    
    results.append(precision[1])
    
mean = sum(results) / len(results)

json = classifier.to_json()

with open('cross_classifier.json', 'w') as json_file:
    json_file.write(json)  
    
classifier.save_weights('cross_classifier.h5')