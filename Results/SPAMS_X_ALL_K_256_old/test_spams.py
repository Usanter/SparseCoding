import spams
import numpy as np
import time
import sys
from sklearn import datasets


# Load dataset
digits = datasets.load_digits()

# Preprocessing
[m,n] = np.shape(digits.data)
#hap = np.transpose(digits.data)
X = digits.data
X = np.asfortranarray(X,dtype=float)


# Spams parameters
param = {'mode':5, 'K':256, 'lambda1':0.0015, 'numThreads':1, 'batchsize':200, 'iter':1000}

#===========First Experiment================

tic = time.time()

# Find a overcomplete dictionary
D= spams.trainDL(X,**param)

# Find sparse linear combination 
H = spams.omp(X,D,lambda1=0.0015) 
tac = time.time()
t = tac - tic

D = np.array(D) #usefull ?

fileD = open('D_spams2.mat','wb')
fileH = open('H_spams2.mat','wb')
np.save(fileD,D)
np.save(fileH,H)
