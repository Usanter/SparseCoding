import spams
import numpy as np
import math
import sys

# Dataset
from tensorflow.examples.tutorials.mnist import input_data


def create_Q(labels, nb_class, k,n):
    plus_one = (k % nb_class) == 0 
    nb_atoms_per_class = int(math.floor(k/nb_class))
    
    # Init Q 
    Q = np.zeros((k,n))
    for item in range(n):
        locations = labels[item].nonzero()
        for place in locations[0]:
            for i in range(nb_atoms_per_class): # Fill
                Q[(nb_atoms_per_class * place)+i,item] = 1 
            if ((place -1 ) == nb_class) and plus_one:
                pass # meh

    return Q

def LC_KSVD(X, Q, lambda1,m, k=1024):

    param = {'mode':5, 'K':1024, 'lambda1':0.04285714285714286, 'numThreads':5, 'batchsize':400, 'iter':100}


    # Find a overcomplete dictionary
    
    D_0 = spams.trainDL(X,**param)
    print("D0........................Done")
    # Find sparse linear combination 
    h_0 = spams.omp(X,D_0,lambda1=0.04285714285714286) 
    print("H0........................Done")
    
    # Compute A_0
    
    lambda2 = 0.5                                   #NOTE Verify this
    I = np.identity(k) 

    A_0 = Q * np.transpose(h_0) * np.linalg.inv(h_0 * np.transpose(h_0) + lambda2 * I)

    print("Initial step..............Done") 

    # Init Ynew and Dnew
    

    Ynew =(np.concatenate([X,math.sqrt(lambda1)*Q]))
    Dnew = (np.concatenate([D_0,math.sqrt(lambda1)*A_0]))
    print("Shape Dnew ",np.shape(Dnew))
    print("Creation Ynew,Dnew........Done")
    
    # Use K-SVD algorithm 
    Ynew = np.asfortranarray(Ynew,dtype=float)   
    Dnew = np.asfortranarray(Dnew,dtype=float)    
    
    D = spams.trainDL(Ynew,D=Dnew,**param)
   
    print("KSVD......................Done")

    # Extract (D and A) and Normalize them 
    D_ext = D[:m] 
    A_ext = D[:n]
    
    Dt = np.transpose(D_ext)
    At = np.transpose(A_ext)
    for i in range(k): # <=> len(At) = len(Dt)
        Dt[i] = Dt[i] / np.linalg.norm(Dt[i],2)
        At[i] = At[i] /np.linalg.norm(Dt[i],2)
    D_hat = np.transpose(Dt)
    A_hat = np.transpose(At)

    # Use OMP algorithm to find h

    D_hat = np.asfortranarray(D_hat,dtype=float)   
    h_hat = spams.omp(X,D_hat,lambda1= lambda1)  

    return D_hat,A_hat,h_hat

#============================================MAIN===============================================
k = 1024
# Load dataset

mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
digits = mnist.train.images[:2000]
digits_labels = mnist.train.labels[:2000]

# Preprocessing

[n,m] = np.shape(digits.data)
X = np.transpose(digits.data)
X = np.asfortranarray(X,dtype=float)

# Define lambda 1 
lambda1 = 1.2/math.sqrt(m)

# Define Q
#Q = np.ones((k,n))
#labels = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
Q = create_Q(digits_labels,len(digits_labels[0]),k,n)
[D,A,h] = LC_KSVD(X,Q,lambda1,m,k)



# Classifier 
