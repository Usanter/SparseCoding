import spams
import numpy as np
import time
import sys
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


# Spams parameters
param = {'mode':5, 'K':265, 'lambda1':0.015, 'numThreads':5, 'batchsize':400, 'iter':2000}

#===========First Experiment================

#tic = time.time()

# Find a overcomplete dictionary
D= spams.trainDL(X,**param)

# Find sparse linear combination 
H = spams.omp(X,D,lambda1=0.015) 
#tac = time.time()
#t = tac - tic

#D = np.array(D) #usefull ?

fileD = open('D_spams4.mat','wb')
fileH = open('h_spams4.mat','wb')
np.save(fileD,D)
np.save(fileH,H)
