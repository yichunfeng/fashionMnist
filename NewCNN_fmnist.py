#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:38:42 2019

@author: yi-chun
"""
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import load_model


fashion_mnist = keras.datasets.fashion_mnist
(train_i, train_l), (test_images, test_labels) = fashion_mnist.load_data()

train_images=np.load('new_image.npy')
train_labels=np.load('new_label.npy')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images.reshape(-1,28, 28,1)/255
test_images = test_images.reshape(-1,28, 28,1)/255      # normalize
train_labels = np_utils.to_categorical(train_labels, num_classes=10)
test_labels = np_utils.to_categorical(test_labels, num_classes=10)



model = Sequential()
# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
batch_input_shape=(None, 28, 28,1),
filters=32,
kernel_size=5,
strides=1,
padding='same',     # Padding method
data_format='channels_first',
))
model.add(Activation('relu'))
# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
pool_size=2,
strides=2,
padding='same',    # Padding method
data_format='channels_first',
))
# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
# Another way to define your optimizer
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
loss='categorical_crossentropy',
metrics=['accuracy'])
print('Training ------------')
# Another way to train the model
model.fit(train_images, train_labels, epochs=70, batch_size=64,)
print('\nTesting ------------')
model.save('new_CNN.h5')
#print('predict before save:',model.predict(test_images)[0])
loss, accuracy = model.evaluate(test_images, test_labels)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


"""for i in range(2000):
    prediction[i]=np.argmax(model.predict(test_images[i]))
    if prediction[i] == test_labels[i]:
        print(prediction[i])
        
"""