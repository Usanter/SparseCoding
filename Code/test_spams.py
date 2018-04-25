import spams
import numpy as np
import time
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data


# Load dataset
mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
digits = mnist.train.images
#digits = datasets.load_digits()

# Preprocessing
[m,n] = np.shape(digits.data)
print([m,n])
#hap = np.transpose(digits.data)
X = np.transpose(digits.data)
X = np.asfortranarray(X,dtype=float)


lambda1 = 1.2/math.sqrt(m)
# Spams parameters
param = {'mode':5, 'K':1024, 'lambda1':lambda1, 'numThreads':5, 'batchsize':400, 'iter':1000}

#===========First Experiment================

#tic = time.time()

# Find a overcomplete dictionary
D= spams.trainDL(X,**param)

# Find sparse linear combination 
H = spams.omp(X,D,lambda1) 
#tac = time.time()
#t = tac - tic

#D = np.array(D) #usefull ?

fileD = open('D_spams_lambdaDyn.mat','wb')
fileH = open('h_spams4_lambdaDyn.mat','wb')
np.save(fileD,D)
np.save(fileH,H)
