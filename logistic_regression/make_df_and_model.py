'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ! Get dataset
filepath = 'csv_with_columns-HWAM.csv'
df = pd.read_csv(filepath)


# ! Make the X and Y data frames
def make_df_for_model(df, X_list, Y_list):
    # Split dataframe into won a medal and didnt win a medal
    df_0 = df[df.MedalEarned == 0]
    df_1 = df[df.MedalEarned == 1]

    # add random number between 0 and 1 to all id
    # Make sure there are same amount of 1 and 0
    df_Bonk = pd.DataFrame(np.random.random(size = (len(df_0), 1)), columns= ['Bonk'])
    df_0_Bonk = df_0.join(df_Bonk)
    
    # Drop df_0_Bonk to approx size of df_1
    dims = (len(df_1) / len(df_0_Bonk))
    df_0_even = df_0_Bonk[df_0_Bonk.Bonk <= dims]

    # concatinate df_0_even and df_1
    dfs = [df_0_even, df_1]
    df_even = pd.concat(dfs)

    # Make test and train dataframes
    for i, row in df_even.iterrows():
        newVal = random.random()

        df_even.at[i,'Bonk'] = newVal

    df_test = df_even[df_even.Bonk < 0.25]
    df_train = df_even[df_even.Bonk >= 0.25]

    # Reduce and split X and Y dataframes
    X_train = df_train[X_list]
    Y_train = df_train[Y_list]
    X_test = df_test[X_list]
    Y_test = df_test[Y_list]
    
    # Create csv files
    X_train.to_csv('X_train.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)


# ! Test X and Y shapes (prints to varify)
def test(X_train, Y_train, X_test, Y_test):
    print("Shape of X_train : ", X_train.shape)
    print("Shape of Y_train : ", Y_train.shape)
    print("Shape of X_test : ", X_test.shape)
    print("Shape of Y_test : ", Y_test.shape)
    print('')
    

# ! sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# ! the model
def model(X, Y, learning_rate, iterations):
    m = X.shape[1] # Observations
    n = X.shape[0] # Types of parameters
    
    W = np.zeros((n,1)) # All a parameters
    B = 0
    
    cost_list = [] # Empty cost list
    
    for i in range(iterations):
        lin_func = np.dot(W.T, X) + B # Linear function
        sig_func = sigmoid(lin_func) # Sigmoid function
        
        # Cost function
        cost = -(1/m)*np.sum( Y*np.log(sig_func) + (1-Y)*np.log(1-sig_func))
        
        # Gradient Descent
        dW = (1/m)*np.dot(sig_func-Y, X.T)
        dB = (1/m)*np.sum(sig_func - Y)
        
        W = W - learning_rate * dW.T
        B = B - learning_rate * dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if False:
            if(i % (iterations / 20) == 0):
                print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list


# ! accuracy test
def accuracy(X, Y, W, B):
    lin_func = np.dot(W.T, X) + B # linear function
    sig_func = sigmoid(lin_func) # Sigmoid function
    
    sig_func = sig_func > 0.5 # Sets sig_func to one if > 0 or 0 if < 0
    
    # Make sig_func array with data type int64
    sig_func = np.array(sig_func, dtype = 'int64') 
    
    # Calculate accuracy
    acc = (1 - np.sum(np.absolute(sig_func - Y)) / Y.shape[1]) * 100
    
    if False:
        print("Accuracy of the model is : ", round(acc, 2), "%")
    
    return acc


# ! run model
def run_model(iterations, learning_rate, test, plot):
    #import training data as pandas dataframe
    X_train = pd.read_csv("X_train.csv")
    Y_train = pd.read_csv("Y_train.csv")

    #import testing data as pandas dataframe
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv")
    
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
    
    #Test dataframes
    if test:
        test(X_train, Y_train, X_test, Y_test)

    W, B, cost_list = model(X_train, Y_train, learning_rate, iterations)
    
    acc = accuracy(X_test, Y_test, W, B)
    
    if plot:
        plt.plot(np.arange(iterations), cost_list)
        plt.show()
    
    return acc


# ! Run multiple iterations of the model
def run_more(times, iterations, learning_rate, test= False, plot= False):
    acc_list = []
    
    for i in range(times):
        # Make X_train, Y_train, X_test, Y_test
        make_df_for_model(df, X_list, Y_list)

        # Run model
        acc = run_model(iterations, learning_rate, test, plot)
        
        acc_list.append(acc)
        
        if len(acc_list) % 5 == 0:
            print(f'on iteration {len(acc_list)} now and still going strong!!!')
    
    acc_avg = sum(acc_list) / len(acc_list)
    acc_min = min(acc_list)
    acc_max = max(acc_list)
    
    print(f'the average accuracy of the model over {times} iterations is: ', round(acc_avg, 2), '%')
    print(f'the lowest accuracy of the model over {times} iterations is', round(acc_min, 2), '%')
    print(f'the highest accuracy of the model over {times} iterations is', round(acc_max, 2), '%')


# ! Variable list for X and Y
X_list = ['ID', 'Height_div_avg', 'Weight_div_avg', 'Age_div_avg']
Y_list = ['ID', 'MedalEarned']

run_more(times = 50, iterations= 3500, learning_rate= 0.0002)