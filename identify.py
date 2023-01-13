import numpy as np
import model as md
import database as img

#first convert input image to an array, using the function to convert input image to array, from get_images.py
def pill_predict(image: np.array):
    """ 
    Function that takes in an image of a pill and returns the model's prediction
    
    Input: np.array "imgs.npy"
    
    input_model(image.shape[0])
    
    Output: string or list of strings of most likely pill OR list of probabilities for each of 5 pill categories
    """
    input_model=md.model
    input_model(image.shape[0])
    return 


    

    
