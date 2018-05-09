from pylab import *
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
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

male_actors = ["Alec Baldwin", "Bill Hader", "Steve Carell"] 
female_actors = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"] 

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

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    """From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/"""

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def main(gender):
    """Main function""" 

    #Create directories to store uncropped/cropped images
    dirs = ["uncropped", "cropped"]
    for direct in dirs:
        if not os.path.exists(direct):
            os.makedirs(direct) 

    if gender == "female":
        SRC_FILE="facescrub_actresses.txt"
        ACTORS=female_actors
    elif gender == "male":
        SRC_FILE="facescrub_actors.txt"
        ACTORS=male_actors
    else: 
       print("Need to pass female|male") 
       sys.exit(2)

    testfile = urllib.URLopener()            

    for a in ACTORS:
        name = a.split()[1].lower()
        i = 0
        for line in open(SRC_FILE):
            if a in line: 
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                face_coords = list(int(j) for j in line.split()[5].split(','))
                ret = timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if (ret != None) and ret:
                    try:
                        raw_image = imread("uncropped/"+filename)
                        face = raw_image[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
                        gray_scale_im = rgb2gray(face) 
                        final_im = imresize(gray_scale_im, size=(32,32,3)) 
                        im = toimage(final_im)
                        im.save("cropped/"+filename)
                        i += 1
                    except:
                        print("Couldn't fetch image, but we shall procceed anyway...")

if __name__ == "__main__":
    main()
