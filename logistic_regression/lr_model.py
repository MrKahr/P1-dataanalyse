import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, diff

#import training data as pandas dataframe
X_train = pd.read_csv("train_X.csv")
Y_train = pd.read_csv("train_Y.csv")

#import testing data as pandas dataframe
X_test = pd.read_csv("test_X.csv")
Y_test = pd.read_csv("test_Y.csv")

# drop id column from dataframes
X_train = X_train.drop("Id", axis = 1)
Y_train = Y_train.drop("Id", axis = 1)
X_test = X_test.drop("Id", axis = 1)
Y_test = Y_test.drop("Id", axis = 1)

# define training and testing dataframes as variables
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

# reshape dataframes to appropriate shape
X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])

# Test
def test():
    print("Shape of X_train : ", X_train.shape)
    print("Shape of Y_train : ", Y_train.shape)
    print("Shape of X_test : ", X_test.shape)
    print("Shape of Y_test : ", Y_test.shape)
    

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def model(X, Y, learning_rate, iterations):
    
    m = X_train.shape[1]
    n = X_train.shape[0]
    
    W = np.zeros((n,1))
    B = 0
    
    cost_list = []
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        # cost function
        cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
        
        # Gradient Descent
        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)
        
        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list