import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# ! Make the X and Y data frames
def make_df_for_model(df, X_list, Y_list):
    # Make MedalValue column
    conditions = [(df['Medal'] == 'Gold'),
                (df['Medal'] == 'Silver'),
                (df['Medal'] == 'Bronze'),
                (df['Medal'] == 'NA')]

    values = [1,1,1,0]

    df['MedalValue'] = np.select(conditions,values)

    # Split dataframe into won a medal and didnt win a medal
    df_0 = df[(df.MedalValue == 0)]
    df_1 = df[(df.MedalValue == 1)]

    # add random number between 0 and 1 to all id
    # Make sure there are same amount of 1 and 0
    for i, row in df_0.iterrows():
        newVal = random.random()

        df_0.at[i,'Bonk'] = newVal

    # Drop df_0 to approx size of df_1
    dims = (len(df_1) / len(df_0) )

    print(f'length of df_1: {len(df_1)}')
    print(f'length of df_0: {len(df_0)}')
    df_0 = df_0.drop(df_0[df_0.Bonk > dims].index)
    print(f'length of df_0 after reduction: {len(df_0)}')

    # concatinate df_0 and df_1
    dfs = [df_0, df_1]
    df = pd.concat(dfs)

    # Make test and train dataframes
    for i, row in df.iterrows():
        newVal = random.random()

        df.at[i,'Bonk'] = newVal

    df_test = df[(df.Bonk < 0.25)]
    df_train = df[(df.Bonk >= 0.25)]

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
    
    print("Accuracy of the model is : ", round(acc, 2), "%")


# ! run model
def run_model(iterations, learning_rate, test):
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
    
    accuracy(X_test, Y_test, W, B)
    
    plt.plot(np.arange(iterations), cost_list)
    plt.show()


# Get dataset
filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

# ! SHD dataframe
if False:
    # Reduce dataframe
    df_a = df[(df.Sex == 'M') & 
            (df.Height > 130) &
            (df.Age > 1) &
            (df.Weight > 1) & 
            (df.Event == 'Athletics Men\'s Shot Put')]

    df_b = df[(df.Sex == 'M') & 
            (df.Height > 130) &
            (df.Age > 1) &
            (df.Weight > 1) & 
            (df.Event == 'Athletics Men\'s Hammer Throw')]

    df_c = df[(df.Sex == 'M') & 
            (df.Height > 130) &
            (df.Age > 1) &
            (df.Weight > 1) & 
            (df.Event == 'Athletics Men\'s Discus Throw')]

    # Concatinate dataframes
    dfss = [df_a, df_b, df_c]
    df = pd.concat(dfss)


# ! Atheletics dataframe
if True:
    print(f'length of df: {len(df)}')
    
    df = df[(df.Sex == 'M') & 
            (df.Height > 130) &
            (df.Age > 1) &
            (df.Weight > 1) & 
            (df.Year > 1945) &
            (df.Sport == 'Athletics')
            ]
    
    print(f'length of df of Athletics: {len(df)}')

# Variable list for X and Y
X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']

# Make X_train, Y_train, X_test, Y_test
make_df_for_model(df, X_list, Y_list)

# Run model
run_model(iterations=3500, learning_rate=0.0002, test=False)