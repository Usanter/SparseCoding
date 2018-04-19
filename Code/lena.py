import spams
import numpy as np
import time
import sys
#from sklearn import datasets
import scipy.misc
import matplotlib.pyplot as plt


# Transform an image into batchs of 16x16
def create_batch(img):
    size = 16
    [img_size_x,img_size_y] = np.shape(img)
    x = 0
    y = 0
    batchs = np.array([])
    while(x+16 <= img_size_x and y+16 <= img_size_y):
        temp = np.array(img[x:x+size,y:y+size])

        batchs = np.append(batchs,np.array([img[x:x+size,y:y+size].reshape((size*size))]))
        x = x + 16
        print("[x]",x)
        if(x+16 > img_size_x):
            y = y +16
            x = 0
            print("[Y]",y)
    return batchs.reshape((int(img_size_x/size)*int(img_size_y/size),size*size))

# Load dataset
lenna =  plt.imread('lenna.tif')

# Preprocessing
[m,n] = np.shape(lenna)
I = np.array(lenna) / 255.
X = np.transpose(create_batch(I))
X = np.asfortranarray(X,dtype=float)


# Spams parameters
param = {'mode':5, 'K':300, 'lambda1':0.015, 'numThreads':4, 'batchsize':100, 'iter':1000}

#===========First Experiment================

tic = time.time()
#foo = create_batch(I)
# Find a overcomplete dictionary
D= spams.trainDL(X,**param)

# Find sparse linear combination 
H = spams.omp(X,D,lambda1=0.015) 
tac = time.time()
t = tac - tic

D = np.array(D) #usefull ?

fileD = open('D_spams_lena.mat','wb')
fileH = open('H_spams_lena.mat','wb')
np.save(fileD,D)
np.save(fileH,H)
