from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import cPickle
from scipy.misc.pilutil import imread
from scipy.misc.pilutil import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import os
from scipy.io import loadmat

np.random.seed(100)

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Part 1: Generate 10 images of each digit
def part1():
    for i in range(10):
        f, axarr = plt.subplots(5, 2, sharex='col', sharey='row')
        axarr[0,0].imshow(M["train"+str(i)][i+1].reshape((28,28)), cmap=cm.gray)
        axarr[0,1].imshow(M["train"+str(i)][i+2].reshape((28,28)), cmap=cm.gray)
        axarr[1,0].imshow(M["train"+str(i)][i+3].reshape((28,28)), cmap=cm.gray)
        axarr[1,1].imshow(M["train"+str(i)][i+4].reshape((28,28)), cmap=cm.gray)
        axarr[2,0].imshow(M["train"+str(i)][i+5].reshape((28,28)), cmap=cm.gray)
        axarr[2,1].imshow(M["train"+str(i)][i+6].reshape((28,28)), cmap=cm.gray)
        axarr[3,0].imshow(M["train"+str(i)][i+7].reshape((28,28)), cmap=cm.gray)
        axarr[3,1].imshow(M["train"+str(i)][i+8].reshape((28,28)), cmap=cm.gray)
        axarr[4,0].imshow(M["train"+str(i)][i+9].reshape((28,28)), cmap=cm.gray)
        axarr[4,1].imshow(M["train"+str(i)][i+10].reshape((28,28)), cmap=cm.gray)
        savefig('images/part1-'+ str(i) + '.png')
#show()

def part3_grad(x, y, output):
    """Returns gradient of weights and bias."""
    # x: the input matrix, y: the target labels, output: the softmax probabilities
    return dot(x, (output-y).T), sum(output-y, axis=1)

# Verify gradient for w
def finite_diffw(x, y_, W0, b0):
    h = np.zeros((784,10))
    step = 1e-9
    
    shape_h = h.shape
    for i in range(shape_h[0]):
        for j in range(shape_h[1]):
            h[i,j] = step 
            L0, f_h = forward(x, W0 + h, b0)
            L1, f_ = forward(x, W0, b0)
            dw, db = part3_grad(x, y_, f_)
            finite = ((NLL(f_h, y_) - NLL(f_, y_))/step)
            print "Finite differences at (%d, %d): %.5f" % (i,j,finite)
            print "Vectorized gradient at (%d, %d): %.5f" % (i,j, dw[i,j])
            h = np.zeros((784,10))

def f(x,y,w,bias):
    """Cost function."""
    L0, output = forward(x, w, bias) 
    return NLL(output, y)
 
# Verify gradient for b
def finite_diffb(x, y_, W0, b0):
    h = np.zeros((10,1))
    step = 1e-9

    shape_h = h.shape
    for i in range(shape_h[0]):
        for j in range(shape_h[1]):
            h[i,j] = step
            L0, f_h = forward(x, W0, b0 + h)
            L1, f_ = forward(x, W0, b0)
            dw, db = part3_grad(x, y_, f_)
            finite = ((NLL(f_h, y_) - NLL(f_, y_))/step)
            print "Finite differences at (%d, %d): %.5f" % (i,j,finite)
            print "Vectorized gradient at (%d, %d): %.5f" % (i,j, db[i])
            h = np.zeros((10,1))

def grad_descent(x, y, init_weights, init_bias, alpha, max_iter, momentum=False):
    # x: your image pixel intensity matrix
    # y: actual labels
    """Run gradient descent."""
    EPS = 1e-10
    prev_t = init_weights-10*EPS
    w = init_weights.copy()
    b = init_bias.copy()
    perf_valid = [] 
    perf_train = [] 
    iterations = [] 
    iter_  = 0
    firstTime = True
    v1 = np.zeros_like(w)
    v2 = np.zeros_like(b)

    #Generate validation/test set
    x_valid , y_valid = make_set(800, "test")

    while iter_ < max_iter and norm(w - prev_t) >  EPS:
        prev_w = w.copy()
        prev_b = b.copy()
        L_, output = forward(x, prev_w, prev_b)
        dw_prev, db_prev = part3_grad(x, y, output)

        L_, output = forward(x, w, b)
        dw, db = part3_grad(x, y, output)
    
        if (firstTime):
            v1 = np.zeros_like(dw)
            v2 = np.zeros_like(db)
            firstTime = False

        if momentum:
            v1 = 0.99*v1 + alpha*dw
            w -= v1          
            v2 = 0.99*v2 + alpha*np.squeeze(array([db]))
            b -= array([v2]).T
        else:
            w -= alpha*np.squeeze(array([dw]))
            b -= alpha*array([db]).T 
            
        #if error decreased, increase learning rate by 10%
        if f(x, y, w, b) < f(x, y, prev_w, prev_b):
            alpha = alpha * 1.1
        else: # error increased. reduce learning rate
            w = prev_w
            b = prev_b
            alpha = alpha * 0.5        

        if iter_ % 1 == 0:
            train_accuracy = getPredictionAccuracy(x,y,w,b)
            valid_accuracy = getPredictionAccuracy(x_valid, y_valid, w, b)
            perf_train.append(train_accuracy)
            perf_valid.append(valid_accuracy)
            iterations.append(iter_)
            
            print "Iter", iter_
            print "Cost %.2f " % (f(x,y,w,b))
        iter_ += 1

    return w, b, iterations, perf_train, perf_valid

def grad_descentP5(x, y, init_weights, init_bias, alpha, max_iter, momentum=False):
    # x: your image pixel intensity matrix
    # y: actual labels
    """Run gradient descent."""
    EPS = 1e-10
    prev_t = init_weights-10*EPS
    w = init_weights.copy()
    b = init_bias.copy()
    perf_valid = [] 
    perf_train = [] 
    iterations = [] 
    iter_  = 0
    firstTime = True
    v1 = np.zeros_like(w)
    v2 = np.zeros_like(b)

    #Generate validation/test set
    x_valid , y_valid = make_set(800, "test")

    while iter_ < max_iter and norm(w - prev_t) >  EPS:
        prev_w = w.copy()
        prev_b = b.copy()
        L_, output = forward(x, prev_w, prev_b)
        dw_prev, db_prev = part3_grad(x, y, output)

        L_, output = forward(x, w, b)
        dw, db = part3_grad(x, y, output)
    
        if (firstTime):
            v1 = np.zeros_like(dw)
            v2 = np.zeros_like(db)
            firstTime = False

        if momentum:
            v1 = 0.99*v1 + alpha*dw
            w -= v1          
            v2 = 0.99*v2 + alpha*np.squeeze(array([db]))
            b -= array([v2]).T
        else:
            w -= alpha*np.squeeze(array([dw]))
            b -= alpha*array([db]).T         

        if iter_ % 1 == 0:
            train_accuracy = getPredictionAccuracy(x,y,w,b)
            valid_accuracy = getPredictionAccuracy(x_valid, y_valid, w, b)
            perf_train.append(train_accuracy)
            perf_valid.append(valid_accuracy)
            iterations.append(iter_)
            
            print "Iter", iter_
            print "Cost %.2f " % (f(x,y,w,b))
        iter_ += 1

    return w, b, iterations, perf_train, perf_valid

def descent(x, y, init_weights, init_bias, alpha, max_iter, momentum=False):
    # x: your image pixel intensity matrix
    # y: actual labels
    EPS = 1e-10
    prev_t = init_weights-10*EPS
    w = init_weights.copy()
    b = init_bias.copy()
    perf_valid = [] 
    perf_train = [] 
    iterations = [] 
    iter_  = 0
    firstTime = True
    v1 = np.zeros_like(w)
    v2 = np.zeros_like(b)

    while iter_ < max_iter and norm(w - prev_t) >  EPS:
        prev_w = w.copy()
        prev_b = b.copy()
        L_, output = forward(x, prev_w, prev_b)
        dw_prev, db_prev = part3_grad(x, y, output)

        L_, output = forward(x, w, b)
        dw, db = part3_grad(x, y, output)
    
        if (firstTime):
            v1 = np.zeros_like(dw)
            v2 = np.zeros_like(db)
            firstTime = False

        if momentum:
            v1 = 0.99*v1 + alpha*dw
            w -= v1          
            v2 = 0.99*v2 + alpha*np.squeeze(array([db]))
            b -= array([v2]).T
        else:
            w -= alpha*np.squeeze(array([dw]))
            b -= alpha*array([db]).T 
            
        #if error decreased, increase learning rate by 10%
        if f(x, y, w, b) < f(x, y, prev_w, prev_b):
            alpha = alpha * 1.1
        else: # error increased. reduce learning rate
            w = prev_w
            b = prev_b
            alpha = alpha * 0.5        

        if iter_ % 1 == 0:
            train_accuracy = getPredictionAccuracy(x,y,w,b)
            perf_train.append(train_accuracy)
            iterations.append(iter_)
            
            print "Iter", iter_
            print "Cost %.2f " % (f(x,y,w,b))
        iter_ += 1
 

    return w, b, iterations, perf_train

def gd_part6(x, y, init_weights, init_bias, alpha, max_iter, i,j, initw_1, init_w2, momentum=False):
    """Run gradient descent."""
    EPS = 1e-10
    prev_t = init_weights-10*EPS
    w = init_weights.copy()
    b = init_bias.copy()
    iter_  = 0
    v1 = np.zeros_like(w)
    v2 = np.zeros_like(b)
    firstTime = True
    
    u1 = 2
    u2 = 6
    #w[i][u1] = 0.7
    #w[j][u2] = -1.7
    w[i][u1] = initw_1
    w[j][u2] = initw_2 
    
    trajectory = [(w[i][u1], w[j][u2])]

    while iter_ < max_iter and norm(w - prev_t) >  EPS:
        prev_w = w.copy()
        prev_b = b.copy()
        
        L_, output = forward(x, w, b)
        dw, db = part3_grad(x, y, output)
        dw_new = np.zeros_like(dw)

        dw_new[i][u1] = dw[i][u1] 
        dw_new[j][u2] = dw[j][u2] 

        if (firstTime):
            v1 = np.zeros_like(dw)
            v2 = np.zeros_like(db)
            firstTime = False


        if momentum:
            v1 = 0.99*v1 + alpha*dw_new
            w -= v1
            #v2 = 0.99*v2 + alpha*np.squeeze(array([db]))
            #b -= array([v2]).T      
            
        else:
            w -= alpha*np.squeeze(array([dw_new]))
            #b -= alpha*array([db]).T



        trajectory.append((w[i][u1], w[j][u2]))
         
        if iter_ % 1 == 0:
            print "Iter", iter_
            print "Cost %.2f " % (f(x,y,w,b))
        iter_ += 1
 
    return w, b, trajectory

def getPredictionAccuracy(x,y,w,b):

    L0, output = forward(x, w, b) # output is 10 by m, each column is probabilities of the 10 classes
    pred = output.copy()
    predArgMax = np.argmax(pred,axis=0) # gets a vector of m elements, each element is the index of the highest probability
    actualY = np.argmax(y, axis=0) # actual response argmax
    result = (predArgMax==actualY) # construct vector of true/false
    #get % correct by summing all "true"s and divide by the # of pts (x.shape[1])
    proportionCorrect = np.sum(result)/(x.shape[1]*1.0)  

    return proportionCorrect

def plot_perfomance(x,y_train, y_valid):
    """Plot learning curve."""

    plt.plot(x, y_train, label="Training Set Accuracy")
    plt.plot(x, y_valid, label="Validation Set Accuracy")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('images/part5-learning.png', bbox_inches='tight')
    show()

def plot_perfomance_part5(x,y_mom,y_no_mom):
    """Plot learning curve."""

    plt.plot(x, y_mom, label="Momentum")
    plt.plot(x, y_no_mom, label="No Momentum")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('images/part5-learning.png', bbox_inches='tight')
    show()

def plot_part6_a(x,y,trained_w, bias,i, j):
    """Plot contour plot of cost fcn."""

    w1s = np.arange(-2, 2, 0.05) #deltas
    w2s = np.arange(-2, 2, 0.05)
    u1 = 2
    u2 = 6  
    w1_orig = trained_w[i][u1]
    w2_orig = trained_w[j][u2]
    C = np.zeros([w1s.size, w2s.size]) 
    new_trained_w = trained_w.copy()

    for m, w1 in enumerate(w1s):
        for n, w2 in enumerate(w2s):
            new_trained_w[i][u1] = w1 + w1_orig
            new_trained_w[j][u2] = w2 + w2_orig
            C[m,n] = f(x,y,new_trained_w,bias) 

    w1s += w1_orig
    w2s += w2_orig
    w1z, w2z = np.meshgrid(w1s, w2s)    
    CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
    plt.xlabel('w1')        
    plt.ylabel('w2')  
    plt.title('Contour plot (Part 6.a)')
    plt.savefig('images/part6_a.png', bbox_inches='tight')
    show()

def plot_part6_d(x,y,trained_w, bias,i, j, traj_gd, traj_mo):
    """Plot trajectory and cost curve."""
    u1 = 2
    u2 = 6
    #w1s = np.arange(0, 2, 0.05) #deltas
    #w2s = np.arange(-1, 1, 0.05)
    w1s = np.arange(-2, 2, 0.05) #deltas
    w2s = np.arange(-2, 2, 0.05)    
    w1_orig = trained_w[i][u1]
    w2_orig = trained_w[j][u2]
    traj = [(trained_w[i][5], trained_w[j][5])]
    C = np.zeros([w1s.size, w2s.size])
    new_trained_w = trained_w.copy()

    for m, w1 in enumerate(w1s):
        for n, w2 in enumerate(w2s):
            new_trained_w[i][u1] = w1 + w1_orig
            new_trained_w[j][u2] = w2 + w2_orig
            C[m,n] = f(x,y,new_trained_w,bias)

    w1s += w1_orig
    w2s += w2_orig
    w1z, w2z = np.meshgrid(w1s, w2s) 
    CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
    plt.xlabel('w1')        
    plt.ylabel('w2')     
    plt.plot([a for a, b in traj_gd], [b for a,b in traj_gd], 'yo-', label="No Momentum")
    plt.plot([a for a, b in traj_mo], [b for a,b in traj_mo], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.title('Contour plot') 
    plt.savefig('images/part6-d.png', bbox_inches='tight')
    show()

def plot_part6_e(x,y,trained_w, bias,i, j, traj_gd, traj_mo):
    """Plot trajectory and cost curve."""
    u1 = 2
    u2 = 6
    #w1s = np.arange(0, 2, 0.05) #deltas
    #w2s = np.arange(-1, 1, 0.05)
    w1s = np.arange(-2, 2, 0.05) #deltas
    w2s = np.arange(-2, 2, 0.05)    
    w1_orig = trained_w[i][u1]
    w2_orig = trained_w[j][u2]
    traj = [(trained_w[i][5], trained_w[j][5])]
    C = np.zeros([w1s.size, w2s.size])
    new_trained_w = trained_w.copy()

    for m, w1 in enumerate(w1s):
        for n, w2 in enumerate(w2s):
            new_trained_w[i][u1] = w1 + w1_orig
            new_trained_w[j][u2] = w2 + w2_orig
            C[m,n] = f(x,y,new_trained_w,bias)

    w1s += w1_orig
    w2s += w2_orig
    w1z, w2z = np.meshgrid(w1s, w2s) 
    CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
    plt.xlabel('w1')        
    plt.ylabel('w2')     
    plt.plot([a for a, b in traj_gd], [b for a,b in traj_gd], 'yo-', label="No Momentum")
    plt.plot([a for a, b in traj_mo], [b for a,b in traj_mo], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.title('Contour plot') 
    plt.savefig('images/part6-e.png', bbox_inches='tight')
    show()

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''

    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def forward(x, W0, b0):
    #b0, biases: 10 by 1, x, input matrix: 784 by m, W0, weights: 784 by 10
    L0 = dot(W0.T, x) + b0
    output = softmax(L0)

    return L0, output #output is the probabilities of each class: 10 by m

#Calculates cost    
def NLL(output, y_):
    return -sum(y_*log(output))

def make_set(set_size, set_type):
    """Construct matrix based on set size."""
    #set_size is number of images per digit
    x = np.zeros((784,10*set_size))
    y = np.zeros((10,10*set_size))
    col = 0

    #Construct set 
    for i in range(10):
        for j in range(set_size):
            image = M[set_type + str(i)][j]/255.0
            x[:,col] = image.flatten().T
            col += 1
        y[i, i*set_size:(i+1)*set_size] = 1

    return x,y

# display weights (Part 4)
def displayWeights(w_train):
    for j in range(10):
        plt.subplot(1,10,j+1).axis('off')
        number = w_train[:, j].reshape((28,28))
        imgplot = plt.imshow(number, cmap=cm.gray)
        plt.savefig('images/part4-weights.png', bbox_inches='tight')
    plt.show()

# Initialize weights/biases
W0 = np.random.randn(784,10)
b0 = np.random.randn(10,1)

#Part 2: computing the forward phase
#200 images per digit
#x , y = make_set(200, "train")
#L0, output = forward(x, W0, b0)

#Part 3: finite-difference checking
#finite_diffw(x, y, W0, b0)
#finite_diffb(x, y, W0, b0)

#Prepare training set
x_train , y_train = make_set(400, "train")

#Part 4 Training the network
#w_train, bias_train, iterations, perf_train, perf_valid = grad_descent(x_train, y_train, W0, b0, 1e-5, 300)
#Display weights
#displayWeights(w_train)
#Plot learning curve
#plot_perfomance(iterations,perf_train, perf_valid)

#Part 5: Gradient Descent with Momentum
#alpha = 1e-5
#w_train, bias_train, iters2, perf_train_mom, perf_valid_mom = grad_descentP5(x_train, y_train, W0, b0, alpha, 300, True)
#w_train, bias_train, iters1, perf_train_no_mom, perf_valid_no_mom = grad_descentP5(x_train, y_train, W0, b0, alpha, 300)
## Learning curve for gradient descent with momentum
#plot_perfomance(iters2,perf_train_mom, perf_valid_mom)
## Comparison of accuracy on training set with and without momentum
#plot_perfomance_part5(iters1, perf_train_mom, perf_train_no_mom)

#Part 6 a.)
# unit indices chosen and output units chosen
# the two weights are: (w_i, u1), (w_j, u2)
w_i = 400
w_j = 300
u1 = 2
u2 = 6
alpha = 1e-5

#w_train, bias_train, iters1, perf_train_no_mom = descent(x_train, y_train, W0, b0, alpha, 1100)

#Cache the trained parameters for repeated testing/trial and error
#np.save("w_train", w_train)
#np.save("bias_train", bias_train)

w_train = np.load("w_train.npy")    
bias_train = np.load("bias_train.npy")
# print the trained weights
print "w1: %.5f, w2: %.5f" % (w_train[w_i,u1], w_train[w_j,u2])


# Part 6 a.) plots the cost function contour plot
#plot_part6_a(x_train,y_train,w_train, bias_train, w_i, w_j)

##Part 6 b.) plots trajectory for gradient descent with no momentum (Example where momentum does well)
#initw_1 = 0.7
#initw_2 = -1.7
#w_train_no_mom, bias_no_mom, gd_traj = gd_part6(x_train, y_train, w_train, bias_train, 7e-2, 30, w_i,w_j, initw_1, initw_2)
#plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
#plt.xlabel('w1')
#plt.ylabel('w2') 
#plt.title('Trajectory (No momentum)')
#plt.savefig('images/part6-b.png', bbox_inches='tight')
#show()

###Part 6 c.) plots trajectory with momentum (Example where momentum does well)
#w_train_mom, bias_mom, mo_traj = gd_part6(x_train, y_train, w_train, bias_train,  5e-2, 13, w_i,w_j, initw_1, initw_2, True)
#plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'yo-', label="Momentum")
#plt.xlabel('w1')        
#plt.ylabel('w2') 
#plt.title('Trajectory (With momentum)')
#plt.savefig('images/part6-c.png', bbox_inches='tight')
#show()

### Part 6 Contour plot + Trajectory (Example where momentum does well)
#plot_part6_d(x_train,y_train,w_train, bias_train,w_i, w_j, gd_traj, mo_traj)

#Part 6 e.) Example where momentum does poorly
initw_1 = -1
initw_2 = 1
w_train_no_mom, bias_no_mom, gd_traj = gd_part6(x_train, y_train, w_train, bias_train, 5e-2, 15, w_i,w_j, initw_1, initw_2)
w_train_mom, bias_mom, mo_traj = gd_part6(x_train, y_train, w_train, bias_train,  5e-2, 15, w_i,w_j, initw_1, initw_2, True)
plot_part6_e(x_train,y_train,w_train, bias_train,w_i, w_j, gd_traj, mo_traj)
