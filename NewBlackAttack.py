#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:34:26 2019

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
import torch


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


file_to_read = open('correct_1k.pkl', 'rb')
data = pickle.load(file_to_read)
print(type(data))
black_i=data[0]
black_l=data[1]
black_image = black_i.reshape(-1,28, 28,1)/255
black_label = black_l.reshape(1000,10)
target=np.zeros([1000,])
black_original=black_i.reshape(-1,28, 28,1)/255

# construct target label
for i in range(1000):
    
        
    if np.argmax(black_label[i])<9:#y_hat
        targets=np.argmax(black_label[i])+1
        
    else:          
        targets=0
    target[i]=targets


np.save('new_black_target.npy',target)  
target= np.load('new_black_target.npy')
black_y=to_categorical(target,num_classes=10)




#x,label = np.load("data_hat.npz")
model = load_model('new_CNN.h5')
model.fit(black_image, black_y, epochs=1, steps_per_epoch=1)
for i in range(1):
    #model = load_model('fmnist_CNN.h5')
    
    loss=K.categorical_crossentropy(black_y,model.output)
    gra=K.gradients(loss,model.input)[0]
    black_image-=0.02*gra
#np.save('black_image.npy',black_image)
#np.save('x_hat.npy',x)




black_image=np.load('black_image.npy')
model = load_model('new_CNN.h5')
b_output=model.predict(black_original,steps=1)
label_original=np.argmax(b_output,1)
adv_output=model.predict(black_image,steps=1)
label_adv=np.argmax(adv_output,1)
index=np.array(np.where(label_adv==target))[0,:]
loss_adv,acc_adv=model.evaluate(black_image,black_y)
print('attacking acc:%.4f.'%(acc_adv))
#np.save('label_adv.npy',label_adv)
#np.save('label_original.npy',label_original)
np.save('new_accuracy_BlackAttack_Adverserial.npy',acc_adv)
image=black_image[index[:],:,:,:].copy()
labels=label_adv[index[:]].copy()
original=black_original[index[:],:,:,:].copy()
num=[]
i=0
while i<10:
    num.append(randint(0,len(index)-1))
    i+=1
    
num_index=index[num]
show_images=image[num,:,:,:]
predict_labels=labels[num]
label_ori=label_original[num_index]

show_images2=original[num,:,:,:]

for i in range(10):
    #plt.subplot(2,5,i+1)
    #plt.grid(False)
    #plt.imshow(show_images2[i,:,:,0])
    plt.subplot(2,5,i+1)
    plt.grid(False)
    plt.imshow(show_images[i,:,:,0])

cla=[]
for i in range(len(predict_labels)):
    i= predict_labels[i]
    cla.append(class_names[i])
print(cla)

