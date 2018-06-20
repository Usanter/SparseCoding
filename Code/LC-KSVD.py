import spams
import numpy as np
import math
import sys
#import matplotlib.pyplot as plt

# Dataset
from tensorflow.examples.tutorials.mnist import input_data
# Classifier
from sklearn.cluster import KMeans


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

    param = {'mode':5, 'K':1024, 'lambda1':0.04285714285714286, 'numThreads':5, 'batchsize':400, 'iter':1000}


    # Find a overcomplete dictionary
    
    D_0 = spams.trainDL(X,**param)
    print("D0........................Done")
    # Find sparse linear combination 
    h_0 = spams.omp(X,D_0,lambda1=0.04285714285714286) 
    print("H0........................Done")
    
    # Compute A_0 
  
    #print(type(Q))
    #print(type(h_0))
    lambda2 = 0.5                                   #NOTE Verify this
    I = np.identity(k) 
    tmp = np.dot(h_0.toarray(),Q.T)
    #print(tmp.shape)
    tmp2 = np.linalg.inv(np.dot(h_0 ,np.transpose(h_0)) + lambda2 * I)
    #print(tmp2.shape)
    #tmp3 = np.dot(tmp,tmp2)
    A_0 =  np.dot(tmp2,tmp)
    #A_0 = np.dot(np.dot(Q, np.transpose(h_0)),np.linalg.inv(np.dot(h_0 ,np.transpose(h_0)) + lambda2 * I))
    print(np.shape(A_0))
    print("Initial step..............Done") 

    # Init Ynew and Dnews
    

    Ynew =(np.concatenate([X,math.sqrt(5)*Q]))
    Dnew = (np.concatenate([D_0,math.sqrt(5)*A_0]))
   
    print("Creation Ynew,Dnew........Done")
    
    # Use K-SVD algorithm 
    Ynew = np.asfortranarray(Ynew,dtype=float)   
    Dnew = np.asfortranarray(Dnew,dtype=float)    
    
    D = spams.trainDL(Ynew,D=Dnew,**param)
   
    print("KSVD......................Done")

    # Extract (D and A) and Normalize them
    #print(np.shape(D))
    D_ext = D[:m] 
    #print(tmp.shape)
    #print(tmp.shape)
    #print(np.shape(D_ext))
    A_ext = D[m:]
    #print(np.shape(D_ext))
    
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
digits = mnist.train.images
digits_labels = mnist.train.labels

# Preprocessing

[n,m] = np.shape(digits.data)
X = np.transpose(digits.data)
X = np.asfortranarray(X,dtype=float)

# Define lambda 1 
lambda1 = 1.2/math.sqrt(m)

# Define Q
#Q = np.ones((k,n))  # Test 1
#labels = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]) # Test 2
Q = create_Q(digits_labels,len(digits_labels[0]),k,n)
[D,A,h] = LC_KSVD(X,Q,lambda1,m,k)


# Save D, A and h
fileD = open('D_LC-KSVD.mat','wb')
fileH = open('h_LC-KSVD.mat','wb')
fileA = open('A_LC-KSVD.mat','wb')
np.save(fileD,D)
np.save(fileH,h.toarray())
np.save(fileA,A)

print("Save files...............Done")

# Classifier
#test = A * h
#test = np.transpose(test)
#kmean1024 = KMeans(init='k-means++', n_clusters=10, n_init=10)
#pred_1024 = kmean1024.fit_predict(test)
#plt.hist(pred_1024)

#print("Kmeans...................Done")
