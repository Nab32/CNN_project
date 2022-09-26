import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

CATEGORIES=['Apple','Orange','Strawberry']

print("Loading...")

def prepareimage(filepath):
    IMG_SIZE=256
    img_array=cv2.imread(filepath)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)

model=tf.keras.models.load_model('model_2')

prediction=model.predict([prepareimage('testimages/test_2.jpg')])[0]
print(CATEGORIES[np.where(prediction==1)[0][0]])