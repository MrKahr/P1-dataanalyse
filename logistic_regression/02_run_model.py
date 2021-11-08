'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ! Get dataset
filepath = 'df_MPHWA_Athletics_3.csv'
df = pd.read_csv(filepath)

# ! Functions that manipulate dataframes and csv files
# * Import and reshapes X and Y files
def import_and_reshape(switch):
    # Import data as pandas dataframes
    X = pd.read_csv(f'X_{switch}.csv')
    Y = pd.read_csv(f'Y_{switch}.csv')
    
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
def even_df(df):
    # Split dataframe into won a medal and didnt win a medal
    df_1 = df[df.MedalEarned == 1]
    df_0 = df[df.MedalEarned == 0]
    
    # Randomly sample df_0 to size of df_1
    df_0 = df_0.sample(n = len(df_1))
    
    return df_1, df_0


# * Make df_test (X_test and Y_test)
def test_sampler(df, X_list, Y_list):
    # Even out for test
    df_1, df_0 = even_df(df)
    
    # Randomly sample test df_1 and df_0
    df_1_test = df_1.sample(n = 150)
    df_0_test = df_0.sample(n = 150)
    
    # Remove test samples from df_1 and df_0
    df = df.drop(df_1_test.index)
    df_testless = df.drop(df_0_test.index)
    
    # Concat df_1_test and df_0_test
    df_test_list = [df_1_test, df_0_test]
    df_test = pd.concat(df_test_list)
    
    # Reduce and split X and Y dataframes
    X_test = df_test[X_list]
    Y_test = df_test[Y_list]
    
    # Create X_test and Y_test csv
    X_test.to_csv('X_test.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)
    
    return df_testless


# * Make the X and Y data frames
def make_df_for_model(df, X_list, Y_list):
    # Randomly sample df_0 to size of df_1
    df_1, df_0 = even_df(df)

    # Randomly sample validate df_1 and df_0
    df_1_validate = df_1.sample(frac= 0.2)
    df_0_validate = df_0.sample(frac= 0.2)

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
    
    # Create csv files
    X_validate.to_csv('X_validate.csv', index=False)
    Y_validate.to_csv('Y_validate.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)


# ! The functions for the logistic regression model
# * Test X and Y shapes (prints to varify)
def test(X_train, Y_train, X_validate, Y_validate):
    print("Shape of X_train : ", X_train.shape)
    print("Shape of Y_train : ", Y_train.shape)
    print("Shape of X_test : ", X_validate.shape)
    print("Shape of Y_test : ", Y_validate.shape)
    print('')
    

# * Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# * The model
def model(X, Y, learning_rate, iterations, cost_progress= False):
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
        
        if cost_progress:
            if(i % (iterations / 20) == 0):
                print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list


# * Accuracy test
def accuracy(X, Y, W, B):
    lin_func = np.dot(W.T, X) + B # linear function
    sig_func = sigmoid(lin_func) # Sigmoid function
    
    sig_func = sig_func > 0.5 # Sets sig_func to one if > 0 or 0 if < 0
    
    # Make sig_func array with data type int64
    sig_func = np.array(sig_func, dtype = 'int64') 
    
    # Calculate accuracy
    acc = (1 - np.sum(np.absolute(sig_func - Y)) / Y.shape[1]) * 100

    return acc


# ! The functions that run the model and report on the model
# * Run model
def run_model(iterations, learning_rate, plot_print= False, cost_progress= False, Test=False):
    # Import and reshape training and validation dataframes
    X_train, Y_train = import_and_reshape('train')
    X_validate, Y_validate = import_and_reshape('validate')
    
    #Test dataframes
    if Test:
        test(X_train, Y_train, X_validate, Y_validate)

    W, B, cost_list = model(X_train, Y_train, learning_rate, iterations, cost_progress)
    
    acc = accuracy(X_validate, Y_validate, W, B)
    
    if plot_print:
        print("Accuracy of the model is : ", round(acc, 2), "%")
        plt.plot(np.arange(iterations), cost_list)
        plt.show()
    
    return W, B, acc


# * Print accuracy
def print_acc_report(list_of_acc, times, name):
    # Calculate average, min and max accuracy
    acc_avg = sum(list_of_acc) / len(list_of_acc)
    acc_min = min(list_of_acc)
    acc_max = max(list_of_acc)
    
    # Print average, min and max accuracy
    print(f'Average {name} over {times} iterations is: ', round(acc_avg, 2), '%')
    print(f'Lowest {name} over {times} iterations is', round(acc_min, 2), '%')
    print(f'Highest {name} over {times} iterations is', round(acc_max, 2), '%')


# * Run multiple iterations of the model
def run_more(times, iterations, learning_rate, plot_print= False, test= False):
    W_list = []
    B_list = []
    acc_list = []
    test_acc_list = []
    
    # Create test sample
    df_testless = test_sampler(df, X_list, Y_list)
    
    for i in range(times):
        # Make X_train, Y_train, X_validate, Y_validate
        make_df_for_model(df_testless, X_list, Y_list)

        # Run model
        W, B, acc = run_model(iterations, learning_rate, plot_print, test)
        
        # Append parameters and accuracy to lists
        W_list.append(W)
        B_list.append(B)
        acc_list.append(acc)
        
        # Progress bar
        if len(acc_list) % 5 == 0:
            print(f'on iteration {len(acc_list)} now and still going strong!!!')

    # Import and reshape test data
    X_test, Y_test = import_and_reshape('test')
    
    # Test parameters on test data
    for i in range(len(W_list)):
        test_acc = accuracy(X_test, Y_test, W_list[i], B_list[i])
        test_acc_list.append(test_acc)
    
    # Print accuracy reports
    print_acc_report(acc_list, times, 'accuracy')
    print_acc_report(test_acc_list, times, 'test accuracy')


# ! Variable list for X and Y
X_list = ['ID',
          'PreviousMedals', 
          'Height_div_avg', 
          'Weight_div_avg', 
          'Age_div_avg'
          ]

Y_list = ['ID', 'MedalEarned']

#make_df_for_model(df, X_list, Y_list)
#run_model(iterations= 5000, learning_rate= 0.025, plot_print= True, cost_progress= True)
run_more(times = 50, iterations= 5000, learning_rate= 0.025)