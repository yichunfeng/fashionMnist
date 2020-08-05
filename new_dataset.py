#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:36:31 2019

@author: yi-chun
"""
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import load_model
from keras import backend as K
from keras.utils import to_categorical
from random import randint
import pickle

fashion_mnist = keras.datasets.fashion_mnist
(train_i, train_l), (test_i, test_l) = fashion_mnist.load_data()

#train_images = train_i.reshape(-1,28, 28,1)/255
#test_images = test_i.reshape(-1,28, 28,1)/255      # normalize
#train_labels = np_utils.to_categorical(train_l, num_classes=10)
#test_labels = np_utils.to_categorical(test_l, num_classes=10)

x_hat=np.load('x_hat.npy')
label= np.load('y_hat.npy')
x_new=x_hat.reshape(1000,28,28)
new_label=np.zeros(61000,)
new_image=np.zeros(shape=(61000,28,28))
for i in range(len(train_l)):
    new_image[i]=train_i[i,:,:]
    new_label[i]=train_l[i]
for i in range(len(label)):
    new_image[i+60000]=x_new[i,:,:]
    new_label[i+60000]=label[i]
print(type(new_image))
print(new_image.shape)
np.save('new_image.npy',new_image)
np.save('new_label.npy',new_label)