import numpy as np
import random 
import torch
import database as db
import identify as idt
from torch import tensor
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import cross_entropy
#from torch.nn import ReLU
from torch.optim import Adam
softmax = nn.functional.softmax
from pathlib import Path
import pickle
ReLU = nn.functional.relu


class Pill_Model(nn.Module):

    def __init__(self, input_shape): #note the first dim will be 3, for 3 color channels
        """ 
        Initializes layers in  model, and sets them 
        as attributes of model.

        Parameters
        -------------------
        input_shape : List representing image 
        """
        super().__init__()
        # data in shape of N x 256 x 256 x 3 (N = # of photos)
        self.conv1 = Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=(7, 7), stride=1)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1)
        self.conv3 = Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(in_features = 21632, out_features = 5, bias=False)

        for m in (self.conv1, self.conv2, self.conv3, self.dense):   #converts from default weight normalization to glorot(xavier)
            #nn.init.constant_(m.bias,0)
            nn.init.xavier_normal_(m.weight, np.sqrt(2))
        
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Model 
        
        Parameters
        ----------
        x : An array of image descriptor vectors?
        Returns
        -------
        self.maxpool(ReLU(self.conv3(pass3))))
        return self.dense(pass3)
        '''
        # TODO : incorporate more layers if needed

        pass1 = self.batchnorm1(self.maxpool(ReLU(self.conv1(x.float()))))
        pass2 = self.batchnorm2(self.maxpool(ReLU(self.conv2(pass1.float()))))
        pass3 = self.batchnorm3(self.maxpool(ReLU(self.conv3(pass2.float()))))
        #print(pass3.shape)
        pass3 = torch.reshape(pass3, (32, 21632))
        #do sofsoftmax(tmax on the final) result to get the probability distribution
        return softmax(self.dense(pass3))

        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        conv1 + self.conv2..parameters + self.conv3 +.parameters
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        # STUDENT CODE HERE
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.conv3.parameters())

#accuracy function
def accuracy(output, target):
    '''
    Calculates accuracy of model, defined by percentage of classifications equal to truth
    
    Parameters
    ----------
    output (shape M x N) : 

    target (shape M x N) : 
    
    Returns
    -------
    mygrad.Tensor, shape=(200,)
        The model's embedded image vectors
    '''
    return np.mean(np.argmax(output, axis = 1) == target)

#one hot encodings for x classes

def train_model():
    model=Pill_Model((3, 256, 256))

    optim = Adam(model.parameters, weight_decay=14e-3)

    num_epochs = 10#define num_epochs
    #train_data is for all images, pill_name is corresponding label for each pill image
    db.load_images()
    num_imgs = len(db.imgs)

    train_data = np.array(db.imgs[:(num_imgs * 9 // 10), 0])
    train_names = db.imgs[:(num_imgs * 9 // 10), 1]

    test_data = db.imgs[(num_imgs * 9 // 10):, 0]
    test_names = db.imgs[(num_imgs * 9 // 10):, 1]

    batch_size= 32 #define this as well
    for epoch_cnt in range(0,num_epochs):
        idxs = np.arange(len(train_data))  
        np.random.shuffle(idxs)  
        
        for batch_cnt in range(0, len(train_data)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            batch = [train_data[i] for i in batch_indices]  # random batch of our training data
            batch = torch.tensor(np.reshape(np.stack(batch), (32, 3, 256, 256)))
            truth = torch.argmax(torch.tensor(db.get_one_hot(train_names[batch_indices])), dim = 1)
            optim.zero_grad()
            prediction = model(batch) 
            
            loss = cross_entropy(prediction, truth)
            loss.backward()
            optim.step()
            acc = accuracy(prediction.detach().numpy(), truth.detach().numpy())
        for batch_cnt in range(0, len(test_data) // batch_size):
            idxs = np.arange(len(test_data))
            batch_indices = batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]#  get the batch of our **test** data
            batch = [test_data[i] for i in batch_indices]#  get the batch of our **test** labels
            batch = torch.tensor(np.reshape(np.stack(batch), (32, 3, 256, 256)))

            with torch.no_grad():
                # get your model's prediction on the test-batch
                prediction = model(batch)#

                # get the truth values for that test-batch
                truth = torch.argmax(torch.tensor(db.get_one_hot(test_names[batch_indices])), dim = 1)#

                # compute the test accuracy
                acc = accuracy(prediction.detach().numpy(), truth.detach().numpy())#

    """
    model=Pill_Model()

    optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
    losses = []
    loss_cnt = 0    
    num_epochs = 6
    batch_size = 32

    #N images, 416 *416*3
    length_input = img.getimg() # TO DO : define all code in get_images under a getimg() function , represents number of photos
    input_data= 0
    for epoch_cnt in range(0, length_input // 5 * 4):
        train_idxs = np.array(0, length_input // 5 * 4)
        np.random.shuffle(train_idxs)

            for batch_cnt in range(0, len(train_idxs)//batch_size):
                pill_names = train_idxs[batch_cnt * batch_size:(batch_cnt + 1) * batch_size]
                batch = [input_data["images"][i]["id"] for i in pill_names]  #not sure about input_data variable
               
    """


            
            
          


         





