from msilib.schema import Binary
from shutil import ExecError
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
import time


print("YEYEYE")
DATADIR='Fruits'

CATEGORIES=['Apple','Orange','Strawberry']

IMG_SIZE=256

training_data=[]

def create_training_data():
    valueofcategory=0
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        num=[0]*3
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                num[valueofcategory]=1
                training_data.append([new_array,num])
            except Exception as e:
                pass
        valueofcategory+=1

create_training_data()
random.shuffle(training_data)


X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)


X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y=np.array(y)

pickle_out=(open('X.pickle','wb'))
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=(open('y.pickle','wb'))
pickle.dump(y,pickle_out)
pickle_out.close()