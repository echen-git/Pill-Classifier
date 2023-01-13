from tokenize import String
import xlrd
from collections import Counter
import urllib.request
from PIL import Image
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random


imgs = []
def get_images():
    '''
    Uses an excel spreadsheet with website path names and labels to retreive images/labels
    Stores as a global numpy array
    No parameters or returns
    '''
    global imgs
    # Give the location of the file
    loc = ("/Users/aaronyang/Desktop/better_pills.xls")

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # Extracting number of rows
    num_rows = sheet.nrows

    for i in range(1, num_rows):
        url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(i)[2]
        urllib.request.urlretrieve(url, str(i))
        img = Image.open(str(i))
        x_crop_len = min(img.size[0] // 2, img.size[1] // 2) 
        y_crop_len = min(img.size[0] // 2, img.size[1] // 2)
        l = (img.size[0] - x_crop_len) // 2
        r = l + x_crop_len
        t = (img.size[1] - y_crop_len) // 2
        b = t + y_crop_len
        img = img.crop((l, t, r, b))
        img = img.resize((256, 256))
        imgs.append(np.array([np.array(img), sheet.row_values(i)[4]]))
        os.remove(str(i))
    imgs = np.random.shuffle(np.array(imgs))

def save_images():
    '''
    Saves global imgs array (containing image and label data) locally
    No parameters or returns
    '''
    global imgs
    with open('imgs.npy', 'wb') as f:
        np.save(f, imgs)

def load_images():
    '''
    Loads saved imgs array and stores it globally
    No parameters or returns
    '''
    global imgs
    with open('imgs.npy', 'rb') as f:
        imgs = np.load(f, allow_pickle = True)

def get_one_hot(names):
    '''
    Takes in drug name and outputs corresponding 1 hot encoding for that drug
    Parameters
    ----------
    names: String
        Name of the drug
    Returns
    --------
    np.ndarray, size (5,)
        Returns a size (5,) one hot encoding
    '''
    labels = []
    for name in names:
        if (name == "AMARYL 4MG TABLETS"):
            labels.append(np.array([1, 0, 0, 0, 0]))
        elif (name == "DEPAKOTE SPRINKLES 125 MG"):
            labels.append(np.array([0, 1, 0, 0, 0]))
        elif (name == "ALLOPURINOL 300MG TABS"):
            labels.append(np.array([0, 0, 1, 0, 0]))
        elif (name == "BETHANECHOL TAB 10MG"):
            labels.append(np.array([0, 0, 0, 1, 0]))
        elif (name == "AMRIX CAP 30MG" or name == "amrix"):
            labels.append(np.array([0, 0, 0, 0, 1]))
    return np.array(labels)
'''
for i in range(5):
    url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(i)[2]
    urllib.request.urlretrieve(url, str(i))
    img = Image.open(str(i))
    print(img.size)
    x_crop_len = min(img.size[0] // 4, img.size[1] // 3) * 4
    y_crop_len = min(img.size[0] // 4, img.size[1] // 3) * 3

    l = (img.size[0] - x_crop_len) // 2
    r = l + x_crop_len
    b = (img.size[1] - y_crop_len) // 2
    t = b + y_crop_len
    img = img.crop((l, t, r, b))
    img = img.resize((2000, 1500))
    os.remove(str(i))
'''


# End goal is to return a array of tuples where each tuple is in the form (image (numpy array), pill name (string))