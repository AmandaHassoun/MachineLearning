import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import glob
from tempfile import TemporaryFile
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imshow

import torch.nn as nn


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.toLastLayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Linear(4096, num_classes),
        )        
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    def getActivations(self, x):
        x = self.toLastLayer(x)
        return x


def build_dict(actors):
    """Split image_list into 3 sets: training, validation and test
       Input - List of paths to formatted images
       Return - Dictionnary with 3 keys ("training", "validation", "test")
                each containing a list of paths to formatted images 
    """
    sets = dict.fromkeys(actors)
    for i in range(len(actors)): 
        sets[actors[i]] = glob.glob("cropped_color/" + actors[i] + "*")
 
    return sets

def get_sets(image_list):
    """Split image_list into 2 sets: training and test
       Input - List of paths to formatted images
       Return - Dictionnary with 2 keys ("training", "test")
                each containing a list of paths to formatted images 
    """

    sets = dict.fromkeys(["training","test"])
    list_size = len(image_list)
    sets["test"] = image_list[:20]
    sets["training"] = image_list[20:]
    
    return sets

def get_train(actors, model):
    batch_xs = np.zeros((0, 43264))
    batch_y_s = np.zeros( (0, 6))
    j = 0
    
    for k in actors.keys():
        train_set = get_sets(actors[k])
        for i in range(len(train_set["training"])):
            im = imread(train_set["training"][i])[:,:,:3]
            im = imresize(im, size=(227,227,3))
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)
            im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
            activations = model.getActivations(im_v).data.numpy().flatten() #43264 array            
            batch_xs = np.vstack((batch_xs, activations))
            one_hot = np.zeros(6)
            one_hot[j] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
        j += 1
    
    return batch_xs, batch_y_s

def get_test(actors, model):
    batch_xs = np.zeros((0, 43264))
    batch_y_s = np.zeros( (0, 6))
    j = 0
    
    for k in actors.keys():
        train_set = get_sets(actors[k])
        for i in range(len(train_set["test"])):
            im = imread(train_set["test"][i])[:,:,:3]
            im = imresize(im, size=(227,227,3))
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)
            im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
            activations = model.getActivations(im_v).data.numpy().flatten() #43264 array            
            batch_xs = np.vstack((batch_xs, activations))
            one_hot = np.zeros(6)
            one_hot[j] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
        j += 1
    
    return batch_xs, batch_y_s

def plot_perfomance(x ,y_train, y_test):
    """Plot learning curve."""

    plt.plot(x, y_train, label="Training")
    plt.plot(x, y_test, label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('part10-learning.png', bbox_inches='tight')
    

if __name__ == "__main__":
    # model_orig = torchvision.models.alexnet(pretrained=True)
    
    actors = ["baldwin", "carell", "bracco", "gilpin", "harmon", "hader"] 
    M = build_dict(actors)   
    model = MyAlexNet()
    model.eval()
    
    #train_x, train_y = get_train(M, model)
    #test_x, test_y = get_test(M, model)
    
    ##Load data into binary file to reduce execution time
    #np.save("outfile_train_x", train_x)
    #np.save("outfile_train_y", train_y)
    #np.save("outfile_test_x", test_x)
    #np.save("outfile_test_y", test_y)    
    
    train_x = np.load("outfile_train_x.npy")    
    train_y = np.load("outfile_train_y.npy")    
    test_x = np.load("outfile_test_x.npy")
    test_y = np.load("outfile_test_y.npy")
    
    print("done")
    
    dim_x = 43264
    dim_h = 30
    dim_out = 6
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor    
    
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    learning_rate = 6e-3
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
    epochs = 100
    batch_size = 32
    accuracy_train = []
    accuracy_test = []
    iters = []
    
    #Normalized data to improve performance
    torch.nn.init.xavier_normal(model[0].weight.data)
    
    for t in range(epochs):
        for r in range(1,train_x.shape[0]//batch_size + 2):
            if r != train_x.shape[0]//batch_size + 1:
                train_idx = np.random.permutation(range(train_x.shape[0]))[(r-1)*batch_size:r*batch_size]
                x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
                y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)  
        
                y_pred = model(x)
                loss = loss_fn(y_pred, y_classes)
                loss_value = loss.data[0]
                
                model.zero_grad()  # Zero out the previous gradient computation
                loss.backward()    # Compute the gradient
                optimizer.step()   # Make a step
                
        if t % 10 == 0:
            iters.append(t)
            
            x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
            x_train = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
            
            y_pred_test = model(x_test).data.numpy()
            y_pred_train = model(x_train).data.numpy()
            
            test_val = np.mean(np.argmax(y_pred_test, 1) == np.argmax(test_y, 1))
            print "loss: %.4f, iterations: %d"% (loss_value, t)
            accuracy_test.append(test_val)
            print("test")
            print(test_val)
            train_val = np.mean(np.argmax(y_pred_train, 1) == np.argmax(train_y, 1))
            accuracy_train.append(train_val)
            print("training")
            print(train_val)
    
    plot_perfomance(iters ,accuracy_train, accuracy_test)    
        
