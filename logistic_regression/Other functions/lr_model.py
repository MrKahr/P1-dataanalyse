import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import training data as pandas dataframe
X_train = pd.read_csv("X_train_HWAY.csv")
Y_train = pd.read_csv("Y_train_HWAY.csv")



#import testing data as pandas dataframe
X_test = pd.read_csv("X_test_HWAY.csv")
Y_test = pd.read_csv("Y_test_HWAY.csv")

# drop id column from dataframes
X_train = X_train.drop("ID", axis = 1)
Y_train = Y_train.drop("ID", axis = 1)
X_test = X_test.drop("ID", axis = 1)
Y_test = Y_test.drop("ID", axis = 1)

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
        
        if(i%(iterations/100) == 0):
            print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list


def accuracy(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")


def run_model():
    iterations = 3000
    learning_rate = 0.00015
    W, B, cost_list = model(X_train, Y_train, learning_rate = learning_rate, iterations = iterations)
    
    accuracy(X_test, Y_test, W, B)
    
    plt.plot(np.arange(iterations), cost_list)
    plt.show()


run_model()

