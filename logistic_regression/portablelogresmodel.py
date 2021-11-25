'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd

# ! Set seed and seed calling function
#rng = np.random.default_rng(1)


# ! Functions that manipulate dataframes and csv files
# * Reshapes X and Y files
def Reshape(X, Y):
    # Drop id column from dataframes
    X = X.drop("ID", axis = 1)
    Y = Y.drop("ID", axis = 1)
    
    # Define dataframes as variables
    X = X.values
    Y = Y.values
    
    # Reshape dataframes to appropriate shape
    X = X.T
    Y = Y.reshape(1, X.shape[1])
    
    return X, Y


# * Make df even
def EvenDF(df, rng):
    # Split dataframe into won a medal and didnt win a medal
    df_1 = df[df.MedalEarned == 1]
    df_0 = df[df.MedalEarned == 0]
    
    # Randomly sample df_0 to size of df_1
    df_0 = df_0.sample(n = len(df_1), random_state=rng.integers(100000))
    
    return df_1, df_0


# * Make the X and Y data frames
def TrainValidate(df, X_list, Y_list, rng):
    # Randomly sample df_0 to size of df_1
    df_1, df_0 = EvenDF(df, rng)
    
    # Randomly sample validate df_1 and df_0
    df_1_validate = df_1.sample(frac= 0.2, random_state=rng.integers(100000))
    df_0_validate = df_0.sample(frac= 0.2, random_state=rng.integers(100000))
    
    # Remove validation samples from df_1 and df_0
    # The rest of df_1 and df_0 are training
    df_1_train = df_1.drop(df_1_validate.index)
    df_0_train = df_0.drop(df_0_validate.index)
    
    # concatinate training and validation
    df_validate_list = [df_1_validate, df_0_validate]
    df_train_list =    [df_1_train, df_0_train]
    
    df_validate = pd.concat(df_validate_list)
    df_train =    pd.concat(df_train_list)
    
    # Reduce and split X and Y dataframes
    X_validate = df_validate[X_list]
    Y_validate = df_validate[Y_list]
    X_train =    df_train[X_list]
    Y_train =    df_train[Y_list]
    
    return X_train, Y_train, X_validate, Y_validate


# ! The functions for the logistic regression model
# * Sigmoid function
def Sigmoid(x):
    return 1/(1 + np.exp(-x))


# * The model
def Model(X, Y, l_rate, iterations):
    m = X.shape[1] # Observations
    n = X.shape[0] # Types of parameters
    
    W = np.zeros((n,1)) # All a parameters
    B = 0
    
    cost_list = [] # Empty cost list
    
    for i in range(iterations):
        lin_func = np.dot(W.T, X) + B # Linear function
        sf = Sigmoid(lin_func) # Sigmoid function
        
        # Cost function
        cost = -(1/m)*np.sum( Y*np.log(sf) + (1-Y)*np.log(1-sf))
        
        # Gradient Descent
        dW = (1/m)*np.dot(sf-Y, X.T)
        dB = (1/m)*np.sum(sf - Y)
        
        W = W - l_rate * dW.T
        B = B - l_rate * dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
    
    return W, B, cost_list


# ! The functions that run the model
# * Run model
def RunModel(df, rng, iterations, l_rate, X_list, Y_list):
    # Make X_train, Y_train, X_validate, Y_validate
    X_train, Y_train, X_validate, Y_validate = TrainValidate(df, X_list, Y_list, rng)
    
    # Import and reshape training and validation dataframes
    X_train, Y_train = Reshape(X_train, Y_train)
    X_validate, Y_validate = Reshape(X_validate, Y_validate)
    
    # Call Model function
    W, B, cost_list = Model(X_train, Y_train, l_rate, iterations)
    
    return W, B


# * Run multiple iterations of the model
def RunMore(df, X_list, Y_list, rng, times, iterations, l_rate):
    W_list = []
    B_list = []
    
    for i in range(times):
        # Run model
        W, B = RunModel(df, rng, iterations, l_rate, X_list, Y_list)
        
        # Append parameters, accuracy and occurances to lists
        W_list.append(W)
        B_list.append(B)
        
        # Progress bar
        if len(W_list) % 10 == 0:
            print(f'{times - len(W_list)} runs left.')
    
    return W_list, B_list
