from pylab import *
from time import sleep
import glob
import threading
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import toimage 
from scipy.misc import imresize
from scipy import stats
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import fetch_data 

def rgb2gray(rgb):
    """Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def get_sets(image_list):
    """Split image_list into 3 sets: training, validation and test
       Input - List of paths to formatted images
       Return - Dictionnary with 3 keys ("training", "validation", "test")
                each containing a list of paths to formatted images 
    """

    sets = dict.fromkeys(["training", "validation", "test"])  
    sets["training"] = image_list[0:70] 
    sets["validation"] = image_list[70:78] 
    sets["test"] = image_list[78:] 
    
    return sets

#Taken from course examples
def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T,x)) ** 2)

#Taken from course examples
def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

#Taken from course examples
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-8
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000 
    iter  = 0

    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*array([df(x, y, t)]).T
        iter += 1

    return t 

def baldwin_or_carell(set_b, set_c): 
    """Linear regression model to classify images of Baldwin and Carell
       Input - Dictionnary containing training set of Baldwin (set_b) 
               and Carell (set_c) 
       Return - Theta matrix
    """

    y = np.zeros((1,140)) #70 images per actor
    theta = np.zeros((1025,1))
    x = np.zeros((1024,140)) 

    i = 0
    while i < 140:
      if i < 70:
        image = imread(set_b["training"][i]) 
        x[:,i] = image.flatten().T 
        y[0][i] = 0 
      else:
        image = imread(set_c["training"][i-70])
        x[:,i] = image.flatten().T
        y[0][i] = 1 
      i += 1

    #Part 4.a): thetas obtained by using the full training set
    print("Part 4.a): Generating image of thetas using the full training set")
    final_theta = grad_descent(f, df, x, y, theta, 1e-11) 
    thetas_im = reshape(final_theta[1:].T, (32,32))
    imsave("part4a-full.png", thetas_im)
    
    #Part 6.c): comparing finite differences to vectorized gradient
    test_finite_difference(x,y,final_theta)

    return final_theta

def part4_a(set_b, set_c):
    """Linear regression model to classify images of Baldwin and Carell
       Input - Dictionnary containing training set of Baldwin (set_b) 
               and Carell (set_c) 
    """

    y = np.zeros((1,4)) #2 images per actor
    y[0][0:2] = 0 #Baldwin
    y[0][2:] = 1 #Carell
    theta = np.zeros((1025,1))
    x = np.zeros((1024,4))

    i = 0
    while i < 4:
      if i < 2:
        image = imread(set_b["training"][i])
        x[:,i] = image.flatten().T
      else:
        image = imread(set_c["training"][i-2])
        x[:,i] = image.flatten().T
      i += 1

    #Part 4.a): thetas obtained by using only 2 images for the training 
    #set 
    final_theta = grad_descent(f, df, x, y, theta, 1e-10)
    thetas_im = reshape(final_theta[1:].T, (32,32))
    imsave("part4a-2.png", thetas_im)
    
def performance_bald_car(set_b, set_c, theta):
    """Evaluating performance of linear regression model"""

    correct_baldwin = 0
    correct_carell = 0

    for i in range(len(set_b)):
        im = imread(set_b[i])
        x = np.reshape(im, 1024)
        x = np.hstack((array([1]), x))
        if np.dot(x, theta) < 0.5:
          correct_baldwin += 1

    for i in range(len(set_c)):
        im = imread(set_c[i])
        x = np.reshape(im, 1024)    
        x = np.hstack((array([1]), x))
        if np.dot(x, theta) >= 0.5:
          correct_carell += 1
   
    print "Baldwin : %d out of %d" % (correct_baldwin, len(set_b))
    print "Carell : %d out of %d" % (correct_carell, len(set_c))

def male_or_female(set_f, set_m, set_size):
    """Linear regression model to classify images of Baldwin and Carell
       Input - Dictionnary containing training set of Baldwin (set_b) 
               and Carell (set_c) 
       Return - Theta matrix
    """

    y = np.zeros((1,set_size*6))
    theta = np.zeros((1025,1))
    x = np.zeros((1024,set_size*6))

    i = 0
    while i < set_size*6:
      if i < set_size*3:
        image = imread(set_m[i])
        x[:,i] = image.flatten().T
        y[0][i] = 0
      else:
        image = imread(set_f[i-(set_size*3)])
        x[:,i] = image.flatten().T
        y[0][i] = 1
      i += 1

    final_theta = grad_descent(f, df, x, y, theta, 1e-12)

    return final_theta

def performance_gender(set_f, set_m, theta):
    """Evaluating performance of linear regression model"""

    male_correct = 0
    female_correct = 0 

    for i in range(len(set_m)):
        im = imread(set_m[i])
        x = np.reshape(im, 1024)
        x = np.hstack((array([1]), x))
        if np.dot(x, theta) < 0.5:
          male_correct += 1

    for i in range(len(set_f)):
        im = imread(set_f[i])
        x = np.reshape(im, 1024)
        x = np.hstack((array([1]), x))
        if np.dot(x, theta) >= 0.5:
          female_correct += 1

    print "Male : %d out of %d" % (male_correct, len(set_m))
    print "Female : %d out of %d" % (female_correct, len(set_f))

def format_new_actors(filelist):
    """Format and re-size manually cropped images for part 5."""

    for actor in filelist:
        raw_image = imread(actor)
        gray_scale_im = rgb2gray(raw_image)
        final_im = imresize(gray_scale_im, size=(32,32,3))
        im = toimage(final_im)
        im.save(actor)

#Part 6.c)
def cost_function_6(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum(sum((y - dot(theta.T, x)) ** 2))

def vectorized_gradient(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return dot(2*x, (dot(theta.T, x) - y).T) 

#Taken from course examples
def grad_descent_6(f, vectorized_gradient, x, y, init_t, alpha):
    EPS = 1e-9
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0

    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*np.squeeze(array([vectorized_gradient(x, y, t)]))
        iter += 1

    return t

def test_finite_difference(x,y,theta):
    """Compare finite difference to vectorized gradient."""

    h = np.zeros((theta.shape)) 
    shape_theta = theta.shape
    for i in range(shape_theta[0]):
        for j in range(shape_theta[1]):
            h[i,j] = 1e-7
            finite = ((cost_function_6(x, y, theta + h) - cost_function_6(x, y, theta))/h)
            print"Finite differences at (%d, %d): %.5f" % (i,j,finite[i,j])
            print "Vectorized gradient at (%d, %d): %.5f" % (i,j,vectorized_gradient(x, y, theta)[i,j])
            h = np.zeros((theta.shape))

#Part 7: Face recognition
def face_recognition(set_bald, set_c, set_bracco, set_g, set_harmon, set_hader):
    """Build linear regression model to distinguish actors by face.""" 

    y = np.zeros((6,420))
    theta = np.zeros((1025,6))
    x = np.zeros((1024,420))

    for i in range(420):
        if i < 70:
            image = imread(set_bald[i])
            x[:,i] = image.flatten().T
            y[0][i] = 1 
        elif (i >= 70) and (i < 140):
            image = imread(set_c[i-70])
            x[:,i] = image.flatten().T
            y[1][i] = 1
        elif (i >= 140) and (i < 210):
            image = imread(set_bracco[i-140])
            x[:,i] = image.flatten().T
            y[2][i] = 1
        elif (i >= 210) and (i < 280):
            image = imread(set_g[i-210])
            x[:,i] = image.flatten().T
            y[3][i] = 1
        elif (i >= 280) and (i < 350):
            image = imread(set_harmon[i-280])
            x[:,i] = image.flatten().T
            y[4][i] = 1
        else:
            image = imread(set_hader[i-350])
            x[:,i] = image.flatten().T
            y[5][i] = 1

    final_theta = grad_descent_6(f, vectorized_gradient, x, y, theta, 1e-12)

    return final_theta

def performance_face_recognition(set_valid, theta):
    """Evaluating performance of linear regression model"""

    correct = 0 
    for i in range(len(set_valid)):
        im = imread(set_valid[i])
        x = np.reshape(im, 1024)
        x = np.hstack((array([1]), x))
        val = np.dot(x, theta)
        if max(val) == val[np.argmax(val)]: 
            correct += 1

    print "%d out of %d \n" % (correct, len(set_valid))

def faces(theta):
    """Save thetas as images.""" 

    thetas_im = reshape(theta[1:,0].T, (32,32))
    imsave("baldwin.png", thetas_im) 

    thetas_im = reshape(theta[1:,1].T, (32,32))
    imsave("carell.png", thetas_im)

    thetas_im = reshape(theta[1:,2].T, (32,32))
    imsave("bracco.png", thetas_im)

    thetas_im = reshape(theta[1:,3].T, (32,32))
    imsave("gilpin.png", thetas_im)      

    thetas_im = reshape(theta[1:,4].T, (32,32))
    imsave("harmon.png", thetas_im)

    thetas_im = reshape(theta[1:,5].T, (32,32))
    imsave("hader.png", thetas_im) 

if __name__ == "__main__":
    #Format images into desired format (i.e. crop, grayscale, resize) 
    print("Fetching Images")
    print("Actresses \n")
    fetch_data.main("female")
    print("Actors \n")
    fetch_data.main("male")

    filelist_baldwin = glob.glob("cropped/baldwin*")
    filelist_carell = glob.glob("cropped/carell*")
    filelist_bracco = glob.glob("cropped/bracco*")
    filelist_gilpin = glob.glob("cropped/gilpin*")
    filelist_harmon = glob.glob("cropped/harmon*")
    filelist_hader = glob.glob("cropped/hader*")

    #Part 2: Seperating dataset into the 3 sets 
    print("Splitting images into 3 sets \n")
    sleep(5)
    baldwin = get_sets(filelist_baldwin)
    carell = get_sets(filelist_carell)
    bracco = get_sets(filelist_bracco)
    gilpin = get_sets(filelist_gilpin)
    harmon = get_sets(filelist_harmon)
    hader = get_sets(filelist_hader)
    
    #Part3: Classify images (Baldwin vs. Carell)
    #Also computes finite diff vs. vectorized gradient 
    print("Part 3: Classifying images (Baldwin vs. Carell) \n")
    sleep(3)
    t = baldwin_or_carell(baldwin, carell)
    print("Evaluating performance of part 3 \n") 
    print("Training Set")
    performance_bald_car(baldwin["training"], carell["training"],t)
    print("Validation Set")
    performance_bald_car(baldwin["validation"], carell["validation"],t)
    sleep(3)
    #Part 4.a): Generating images using full training set vs. 2 images  
    print("Part 4.a): Generating images using full training set vs. 2 images")
    part4_a(baldwin, carell) 

    #Part 5: performance on different training set sizes
    print("Part 5: Performance on different set sizes \n")
    list_sizes = [10,20,30,40,50,60,70] 
    females_v = bracco["validation"]+ gilpin["validation"] + harmon["validation"]
    males_v = baldwin["validation"] + carell["validation"] + hader["validation"]
    theta = np.zeros((1025,1))  
    for i in range(len(list_sizes)):
        print "SET SIZE: %d \n" %(list_sizes[i])
        list_sizes = [10,20,30,40,50,60,70] 
        females_t = bracco["training"][0:list_sizes[i]] + gilpin["training"][0:list_sizes[i]] + harmon["training"][0:list_sizes[i]]  
        males_t = baldwin["training"][0:list_sizes[i]] + carell["training"][0:list_sizes[i]] + hader["training"][0:list_sizes[i]]
        theta = male_or_female(females_t, males_t, list_sizes[i]) 
        print("Training set performance")
        performance_gender(females_t,males_t, theta)
        print("Validation set performance")
        performance_gender(females_v,males_v, theta)
        sleep(3) 
    
    print("Part 5: Performance on 6 actors not included \n")
    filelist_new_actors = glob.glob("jim*") + glob.glob("jake*")
    filelist_new_actresses = glob.glob("jen*")
    format_new_actors(filelist_new_actors)
    format_new_actors(filelist_new_actresses)
    performance_gender(filelist_new_actresses, filelist_new_actors, theta)

    #Part 7: Facial recognition 
    print("Part 7: Facial Recognition \n")
    t = face_recognition(baldwin["training"], carell["training"], bracco["training"], gilpin["training"], harmon["training"], hader["training"])
    print("Validation set:")
    print("Baldwin")
    performance_face_recognition(baldwin["validation"], t)
    print("Carell")
    performance_face_recognition(carell["validation"], t)
    print("Bracco")
    performance_face_recognition(bracco["validation"], t)
    print("Gilpin")
    performance_face_recognition(gilpin["validation"], t)
    print("Harmon")
    performance_face_recognition(harmon["validation"], t)
    print("Hader")
    performance_face_recognition(hader["validation"], t)
    print("Training set:")
    print("Baldwin")
    performance_face_recognition(baldwin["training"], t)
    print("Carell")
    performance_face_recognition(carell["training"], t)
    print("Bracco")
    performance_face_recognition(bracco["training"], t)
    print("Gilpin")
    performance_face_recognition(gilpin["training"], t)
    print("Harmon")
    performance_face_recognition(harmon["training"], t)
    print("Hader")
    performance_face_recognition(hader["training"], t)
    sleep(3) 

    #Part 8: Saving thetas as images
    print("Part 8: Saving thetas as images")
    faces(t)
