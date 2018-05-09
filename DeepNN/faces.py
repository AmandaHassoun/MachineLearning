from torch.autograd import Variable
import glob
import torch
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

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

def get_train(actors):
    batch_xs = np.zeros((0, 64*64))
    batch_y_s = np.zeros( (0, 6))
    part9_act1 = np.zeros((0, 64*64))
    part9_act2 = np.zeros((0, 64*64))
    
    j = 0
    
    for k in actors.keys():
        train_set = get_sets(actors[k])
        for i in range(len(train_set["training"])):
            if (i < 3) and (k == 'hader'):
                print(k)
                part9_act1 = np.vstack((part9_act1, np.array(imread(train_set["training"][i]).flatten().T/255.)))
            if (i < 3) and (k == 'harmon'):
                print(k)
                part9_act2 = np.vstack((part9_act2, np.array(imread(train_set["training"][i]).flatten().T/255.)))            
      
            batch_xs = np.vstack((batch_xs, np.array(imread(train_set["training"][i]).flatten().T/255.)))
            one_hot = np.zeros(6)
            one_hot[j] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
        j += 1 
    
    return batch_xs, batch_y_s, part9_act1, part9_act2


def get_test(actors):
    batch_xs = np.zeros((0, 64*64))
    batch_y_s = np.zeros( (0, 6))
    j = 0
    
    for k in actors.keys():
        train_set = get_sets(actors[k])
        for i in range(len(train_set["test"])):
            batch_xs = np.vstack((batch_xs, np.array(imread(train_set["test"][i]).flatten().T/255.)))
            one_hot = np.zeros(6)
            one_hot[j] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
        j += 1
    
    return batch_xs, batch_y_s

def build_dict(actors):
    """Split image_list into 3 sets: training, validation and test
       Input - List of paths to formatted images
       Return - Dictionnary with 3 keys ("training", "validation", "test")
                each containing a list of paths to formatted images 
    """
    sets = dict.fromkeys(actors)
    for i in range(len(actors)): 
        sets[actors[i]] = glob.glob("cropped/" + actors[i] + "*")
 
    return sets

def plot_perfomance(x ,y_train, y_test):
    """Plot learning curve."""

    plt.plot(x, y_train, label="Training Set Accuracy")
    plt.plot(x, y_test, label="Test Set Accuracy")
    plt.xlabel('Iteration (Total=1500)')
    plt.ylabel('Accuracy (learning rate = 1e-5)')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('part8-learning.png', bbox_inches='tight')


if __name__ == "__main__":
    np.random.seed(10)
    actors = ["baldwin", "carell", "bracco", "gilpin", "harmon", "hader"] 
    M = build_dict(actors)

    #-------------Part 8------------------
    train_x, train_y, part9_act1, part9_act2 = get_train(M)
    test_x, test_y = get_test(M)
    
    dim_x = 64*64
    dim_h = 18
    dim_out = 6
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor    
    
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 1500
    batch_size = 16
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
                
                model.zero_grad()  # Zero out the previous gradient computation
                loss.backward()    # Compute the gradient
                optimizer.step()   # Use the gradient information to 
                                   # make a step
        if t % 100 == 0:
            iters.append(t)
            
            x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
            x_train = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
            
            y_pred_test = model(x_test).data.numpy()
            y_pred_train = model(x_train).data.numpy()
            
            accuracy_test.append(np.mean(np.argmax(y_pred_test, 1) == np.argmax(test_y, 1)))
            accuracy_train.append(np.mean(np.argmax(y_pred_train, 1) == np.argmax(train_y, 1)))
    
    
    # Plot the learning curve for training/test set.
    #plot_perfomance(iters ,accuracy_train, accuracy_test)
    
    #-------------Part 9----------------
    # Displays the weights of the output units
    output_act1 = np.dot(model[0].weight.data.numpy(), part9_act1.T)
    output_act2 = np.dot(model[0].weight.data.numpy(), part9_act2.T)
    
    #hader: 0, 17
    plt.imshow(model[0].weight.data.numpy()[4, :].reshape((64, 64)), cmap=plt.cm.coolwarm)
    plt.imshow(model[0].weight.data.numpy()[8, :].reshape((64, 64)), cmap=plt.cm.coolwarm)
    
    #harmon: 11, 14
    plt.imshow(model[0].weight.data.numpy()[11, :].reshape((64, 64)), cmap=plt.cm.coolwarm)
    plt.imshow(model[0].weight.data.numpy()[14, :].reshape((64, 64)), cmap=plt.cm.coolwarm)
