from PIL import Image
import os, os.path
import numpy as np
import cv2
import glob

# Images from that weird dataset/pretrained model

imgs = []
def get_images():
    path = "/Users/aaronyang/Desktop/Pill Classification.v2-augmented-classification-v1.folder/train/pill"
    for f in os.listdir(path):
        imgs.append(np.array(Image.open(os.path.join(path,f))))
    imgs = np.random.shuffle(np.array(imgs))

def save_images():
    with open('imgs.npy', 'wb') as f:
        np.save(f, imgs)


def load_images():
    with open('imgs.npy', 'rb') as f:
        imgs = np.load(f)




'''
X_data = []
files = glob.glob ("/Users/aaronyang/Desktop/Pill Classification.v2-augmented-classification-v1.folder/train/pill")
for myFile in files:
    image = cv2.imread (myFile)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape)
'''