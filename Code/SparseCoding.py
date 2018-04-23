#Imports
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from sklearn import datasets

#==============================================
## Dimensions:
## x - R^(m  n)
## D - R^(m  k)
##    with k columns referred to nb of  atoms
##=============================================

#======================MISC FUNCTIONS============================================
# Function which compute the "convergence" about 2 matrix
def Convergence(Matrix_before,Matrix_after):
    sys.stdout.write("\rOpti D " +str(np.sum(abs(Matrix_before - Matrix_after))))
    sys.stdout.flush()
    #print("Diff:",np.sum(bs(Matrix_before - Matrix_after)))
    return np.sum(abs(Matrix_before - Matrix_after)) > 1

def Convergence2(Matrix_before,Matrix_after):
    sys.stdout.write("\rOpti H " + str(np.sum(abs(Matrix_before - Matrix_after))))
    sys.stdout.flush() 
    #print("Diff2:",np.sum(abs(Matrix_before - Matrix_after)))
    return np.sum(abs(Matrix_before - Matrix_after)) > 1


# Function which help to avoid the non-division by 0 in the h's gradient descent
def shrink(h,coef):
    result = np.array([])
    # for all elements sign(hi) * max( |hi| - coef, 0)
    for hi in h:
        #print(np.sign(hi))
        hi = np.sign(hi)* np.max(np.abs(hi) - coef, 0)
        #print(test)
        #np.append(result,np.sign(hi) * np.max(np.abs(hi) - coef,0))
    #print(result.shape)
    return h

# Function for h gradient descent
def ISTA(h,D,x,k,alpha = 0.05,lambda_coef = 0.2):
    
    lambda_coef = 1.2/math.sqrt(len(x)) # Cf Online Dictionary learning for Sparse Coding Doc
    not_converged = True 
    # While h^(t) has not converged
    while(not_converged):
        h_before = np.array(h)
	
        for i in range(k):

            #h^(t) <= h^(t) - alpha D' (D h^(t) -x^(t))

            h[i] = h[i] - alpha * (np.transpose(D[:,i]).dot(D.dot(h) -x))
         
            # h^(t) <= shrink (h^(t), alpha lamdba)
            h[i] = shrink(h[i],alpha * lambda_coef)

        #Check if convergence
        not_converged = Convergence2(h_before,h)

    # return h^(t)
    return h

# Function for D's block coordinate descent
def block_coordinate_descent(D,A,B,k):

    for j in range(k):

        # Computation
        D[:,j] = (1/A[j,j]) * (B[:,j] - (D.dot(A[:,j])) + (D[:,j].dot(A[j,j])))

        # Normalization
        D[:,j] = D[:,j] / (np.linalg.norm(D[:,j]))

    return D

# Function for D's gradient descent #NOTE: Check if there are bugs
def Dictionary_gradient_descent(D,x,k,alpha = 0.5):
    not_converged = True
    # While D has not converged
    while(not_converged):
            D_before = np.aray(D)
            # Compute D = D - alpha  * 1/T Sum(x - Dh * transpose(h))
            D = D - alpha  * sum( (x -Dh)* np.transpose(h))
            # Normalise
            for j in range(k):
               D[:,j] = (D[:,j] / np.norm(D[:,j]))
            # Check convergence
            not_converged = Convergence(D_before,D)

    return D
# Compute current cost 
def compute_cost(x,D,h,lambda_coef = 0.2):

        [_,n] = np.shape(x) 
        return (1/n)* ( (1/2)*np.linalg.norm(x - D.dot(h))**2 + lambda_coef*np.linalg.norm((h),ord=1))

#==============================SPARSE CODING ALGORITHM======================================
# Sparse coding  using Online learning algorithm
def sparse_coding_online(x,k = 265,beta=0.3):#NOTE: Check how to make ISTA with only 1 x_t (dimension pb)
    [m,n] = np.shape(x)
    #Initial dictionary (consisting of unit normm atoms sampled from the unit sphere)
    D = np.random.rand(m,k)

    print("SHAPE ",np.shape(D))

    print("D",D[:,1])
    print("X",np.shape(x))
    #Initial coef
    h = np.zeros((k,n))
    A = np.zeros((k,k))
    B = np.zeros((len(x),k))
    # While  D not_converged
    not_converged = True
    iteration = 1
    cost = np.array(compute_cost(x,D,h))
    fig = plt.figure()
    plt.plot(cost,'r') 
    #plt.hold(True)
    #plt.show()
    for x_t in np.transpose(x):
        print("\nIteration ",iteration)
        print("==============================")
        #Infer code h
        h = ISTA(h,D,x_t,k,0.01)
        print(" ")
        #Update dictionary:
        # B = beta * B + (1 - beta)  * transpose(h)* x
        #B = beta * B + ( 1 - beta) *  x_t.dot(np.transpose(h))
        # A = beta * A + (1 - beta) * transpose(h)* h
        #A = beta * A + (1 - beta) * h.dot(np.transpose(h))
            
        A = A + h.dot(np.transpose(h))
        B = B + x.dot(np.transpose(h))
        # fileA = open('A','wb')    
        #np.save(fileA,A)
        # fileA = open('B','wb')
        #np.save(fileA,B)
          
        # While D not_converged:
        D_not_converged = True
        while(D_not_converged):
            D_before2 = np.array(D)
            for j in range(k):
            # For each column D[:,j]
                    
                # Update for the column j
                D[:,j] = (1/A[j,j]) * (B[:,j] - (D.dot(A[:,j])) + (D[:,j].dot(A[j,j])))

                # Normalize
                D[:,j] = D[:,j] / (np.linalg.norm(D[:,j]))
            D_not_converged = Convergence(D_before2,D)
        iteration = iteration +1
        cost =  np.append(cost,compute_cost(x,D,h))

        plt.plot(cost,'r')
    plt.show()
    return [D,h]   

# Sparse coding using Block-coordinate descent algorithm
def sparse_coding(x,k = 265):

    #Init
    not_converged =  True
    it = 1
    [m,n] = np.shape(x)
    print("[1]  ,",[m,n])
    #x = np.transpose(x) 
   
    #Initial dictionary (consisting of unit normm atoms sampled from the unit sphere)
    D = np.random.rand(m,k)

    #Initial coef
    h = np.zeros((k,n))


    #Print initial cost
    cost = np.array(compute_cost(x,D,h))
    fig = plt.figure()
    plt.plot(cost,'r') 
    # While D has not converged
    while(not_converged):

        D_before = np.array(D)

        print("\n================================================")
        print("\nIteration ",it)
        # Find the sparse code h(x^(t)) for all x^(t) in
        # the training set with ISTA
        h = ISTA(h,D,x,k)

        print("\n")
        # Update the dictonary

        # Computation A
        A = h.dot(np.transpose(h))

        # Computation B
        B = x.dot(np.transpose(h))
        
        # Run block-coordinate descent algorithm to update D
        D_not_converged = True
        while(D_not_converged):
            D_before2 = np.array(D)
            for j in range(k):
            # For each column D[:,j]
                    
                # Update for the column j
                D[:,j] = (1/A[j,j]) * (B[:,j] - (D.dot(A[:,j])) + (D[:,j].dot(A[j,j])))

                # Normalize
                D[:,j] = D[:,j] / (np.linalg.norm(D[:,j]))
            D_not_converged = Convergence(D_before2,D)
        it = it +1
        
        #Check convergence
        not_converged = Convergence(D_before,D)

        cost = np.append(cost, compute_cost(x,D,h))
        plt.plot(cost,'r')
    plt.show()
    return [D,h]

#===================================== MAIN =======================================================

# Load digits for the example
digits = datasets.load_digits()

# Use sparse coding to extract the dictionary and the sparse representation
[D,h] = sparse_coding(np.transpose(digits.data[:100]))
#[D,h] = sparse_coding_online(np.transpose(digits.data[:200])) # We take only 200 data (time saving)


# Save the value of D and h
fileD = open('D.mat','wb')
fileH = open('h.mat','wb')
np.save(fileD,D)
np.save(fileH,h)
k = 20
plt.figure() # to be sure I will not overide previous plot

# Plot the dictionary
size_img = 8 # Because we have 8x8 images (digits)
fig,table = plt.subplots(4,5)
index_x = 0
index_y = 0

for i in range(k):
    img = D[:,i]
    img = np.reshape(img,(size_img,size_img))
    table[index_x,index_y].imshow(img, cmap='gray')
    index_y = index_y + 1
    if index_y > 4:
        index_x = index_x + 1
        index_y = 0
fig.show()
