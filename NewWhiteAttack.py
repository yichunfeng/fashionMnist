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


fashion_mnist = keras.datasets.fashion_mnist
(train_i, train_l), (test_i, test_l) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_i.reshape(-1,28, 28,1)/255
test_images = test_i.reshape(-1,28, 28,1)/255      # normalize
train_labels = np_utils.to_categorical(train_l, num_classes=10)
test_labels = np_utils.to_categorical(test_l, num_classes=10)


model = load_model('new_CNN.h5')
y = model.predict(test_images)
   


x=np.zeros([1000,28,28,1])

label=np.zeros([1000,])

i=0
j=0
# pick the x_hat

for i in range(1000):
    
    if np.argmax(y[i])==np.argmax(test_labels[i]):
        
        if np.argmax(test_labels[i])<9:#y_hat
            #print("before:",np.argmax(test_labels[i]))
            labels=np.argmax(test_labels[i])+1
            #print("label:",label)
        else:
            #print("before:",np.argmax(test_labels[i]))
            labels=0
            #print("label:",label)
           
        x[i,:,:,:]=test_images[j,:,:,:]
        label[i]=labels
    j+=1
    
np.save('new_x_pick.npy',x)  
np.save('new_y_hat.npy',label)   
  

x= np.load('new_x_pick.npy')
label= np.load('new_y_hat.npy')






#x,label = np.load("data_hat.npz")

model = load_model('new_CNN.h5')
for i in range(5):
    
    x_hat = x.reshape(-1,28, 28,1)/255
    y_hat=to_categorical(label,num_classes=10)
    #epsilon=0.3
    #above=x+epsilon
    #below=x-epsilon
    model.fit(x_hat, y_hat, epochs=1, steps_per_epoch=1)
    loss=K.categorical_crossentropy(y_hat,model.output)
    gra=K.gradients(loss,model.input)[0]
    x_hat-=0.02*gra

np.save('new_x_hat.npy',x)

"""
x_hat=np.load('x_hat.npy')
history=model.fit(x_hat, y_hat, epochs=30, batch_size=100,)
acc_adv=history.history['acc']
print(acc_adv)
print(type(acc_adv))
"""


x_hat=np.load('new_x_hat.npy')
model = load_model('new_CNN.h5')
x_output=model.predict(x,steps=1)
label_original=np.argmax(x_output,1)
y_adv=model.predict(x_hat,steps=1)
label_adv=np.argmax(y_adv,1)
index=np.array(np.where(label_adv!=label_original))[0,:]
loss_adv,acc_adv=model.evaluate(x_hat,y_hat)
print('attacking acc:%.4f.'%(acc_adv))
#np.save('label_adv.npy',label_adv)
#np.save('label_original.npy',label_original)
np.save('new_accuracy_WhiteAttack_Adverserial.npy',acc_adv)
whitebox_image=x_hat[index[:],:,:,:].copy()
whitebox_labels=label_adv[index[:]].copy()
original_image=x[index[:],:,:,:].copy()
num=[]
i=0
while i<10:
    num.append(randint(0,len(index)))
    i+=1
    
num_index=index[num]
show_images=whitebox_image[num,:,:,:]
predict_labels=whitebox_labels[num]
label_ori=label_original[num_index]

show_images2=original_image[num,:,:,:]

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

