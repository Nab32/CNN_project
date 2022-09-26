from gc import callbacks
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
import time

X=pickle.load(open('X.pickle','rb'))
y=pickle.load(open('y.pickle','rb'))

X=X/255.0

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=X.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=15,epochs=10,validation_split=0.25)

model.save('model_4')