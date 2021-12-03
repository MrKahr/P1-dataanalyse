'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd


# ! Functions that manipulate dataframes and csv files
# * Reshapes X and Y files
def Reshape(X, Y):
    # Drop id column from dataframes
    X = X.drop("ID", axis = 1)
    Y = Y.drop("ID", axis = 1)
    
    # Define dataframes as variables
    # Return a Numpy representation of the DataFrame.
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


# * Make df_test (X_test and Y_test)
def TestSampler(df, rng, X_list, Y_list):
    # Split dataframe into won a medal and didnt win a medal
    df_1 = df[df.MedalEarned == 1]
    df_0 = df[df.MedalEarned == 0]
    
    # Randomly sample test df_1 and df_0
    df_1_test = df_1.sample(n = 100, random_state=rng.integers(100000))
    df_0_test = df_0.sample(n = 100, random_state=rng.integers(100000))
    
    # Remove test samples from df_1 and df_0
    df = df.drop(df_1_test.index)
    df_testless = df.drop(df_0_test.index)
    
    # Concat df_1_test and df_0_test
    df_test_list = [df_1_test, df_0_test]
    df_test = pd.concat(df_test_list)
    
    # Reduce and split X and Y dataframes
    X_test = df_test[X_list]
    Y_test = df_test[Y_list]
    
    return df_testless, X_test, Y_test


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
def RunModel(df, rng, cop, iterations, l_rate, X_list, Y_list):
    # Make X_train, Y_train, X_validate, Y_validate
    X_train, Y_train, X_validate, Y_validate = TrainValidate(df, X_list, Y_list, rng)
    
    # Import and reshape training and validation dataframes
    X_train, Y_train = Reshape(X_train, Y_train)
    X_validate, Y_validate = Reshape(X_validate, Y_validate)
    
    # Call Model function
    W, B, cost_list = Model(X_train, Y_train, l_rate, iterations)
    
    sf_val = Classify(X_validate, W, B, cop)
    val_acc, val_occ_dic = Accuracy(sf_val, Y_validate)
    
    return W, B, val_acc, val_occ_dic

# * Classify winners and losers
def Classify(X, W, B, cop):
    lin_func = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lin_func) # Sigmoid function
    
    # Make sf binary array with data type int64
    sf = sf > cop # Sets sf to one if > 0 or 0 if < 0
    sf = np.array(sf, dtype = 'int64')
    
    return sf


# * Calculate accuracy of the model
def Accuracy(sf, Y):
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    guesses = sf * 2 - Y
    occurance = [[x, list(guesses[0]).count(x)] for x in set(list(guesses[0]))]
    occ_d = {1:0, 0:0, -1:0, 2:0}
    
    # Assign value to keys e.g. TP : 22
    for i in occurance: 
        occ_d[i[0]] = i[1]
    
    # True Positive, True Negative, False Positive and False Negative
    tp, tn, fp, fn = occ_d[1], occ_d[0], occ_d[2], occ_d[-1]
    
    # Calculate accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc, occ_d


# * Print accuracy
def PrintAccReport(list_of_acc, name):
    # Calculate average, min and max accuracy
    acc_avg = sum(list_of_acc) / len(list_of_acc)
    acc_min = min(list_of_acc)
    acc_max = max(list_of_acc)
    
    # Print average, min and max accuracy
    print(f'{name} results: avg {round(acc_avg * 100, 2)}% ', 
                    f'min {round(acc_min * 100, 2)}% ', 
                    f'max {round(acc_max * 100, 2)}% ')


# * Run multiple iterations of the model
def RunMore(df, X_list, Y_list, rng, cop, times, iterations, l_rate):
    W_list = []
    B_list = []
    val_acc_list = []
    val_occ_list = []
    
    # Create test sample
    df_testless, X_test, Y_test = TestSampler(df, rng, X_list, Y_list)
    
    for i in range(times):
        # Run model
        W, B, val_acc, val_occ_dic = RunModel(df_testless, rng, cop, iterations, l_rate, X_list, Y_list)
        
        # Append parameters, accuracy and occurances to lists
        W_list.append(W)
        B_list.append(B)
        val_acc_list.append(val_acc)
        val_occ_list.append(val_occ_dic)
        
        # Progress bar
        if len(W_list) % 10 == 0:
            print(f'{times - len(W_list)} runs left.')
    
    # Print test sample accuracy
    PrintAccReport(val_acc_list, 'Validation')
    
    X_test, Y_test = Reshape(X_test, Y_test)
    test_occ_dic_list = []
    test_acc_list = []
    
    # Test parameters on test data
    for i in range(len(W_list)):
        sf_test = Classify(X_test, W_list[i], B_list[i], cop)
        test_acc, test_occ_dic = Accuracy(sf_test, Y_test)
        test_acc_list.append(test_acc)
        test_occ_dic_list.append(test_occ_dic)
    
    # Print test sample accuracy
    PrintAccReport(test_acc_list, 'Test sample')
    
    W_array = np.concatenate(W_list, axis=1)
    B_array = np.stack(B_list)
    
    np.savetxt('W.csv', W_array, delimiter= ',')
    np.savetxt('B.csv', B_array, delimiter= ',')


# ! Run parameters on decathlon athletes
def Decathlon(df, X_list, Y_list, W_array, B_array, cop):
    dec_acc_list = []
    dec_occ_list = []
    
    # Reduce and split X and Y dataframes
    X_dec = df[X_list]
    Y_dec = df[Y_list]
    
    # Import and reshape dec data
    X_dec, Y_dec = Reshape(X_dec, Y_dec)
    
    # Test parameters on dec
    for i in range(len(W_array[0])):
        W_par = np.array([W_array[0][i], W_array[1][i], W_array[2][i], W_array[3][i]], ndmin= 0)
        sf = Classify(X_dec, W_par, B_array[i], cop)
        da, dod = Accuracy(sf, Y_dec)
        dec_acc_list.append(da)
        dec_occ_list.append(dod)
    
    # Print decathlon accuracy
    PrintAccReport(dec_acc_list, 'Decathlon')
    
    return dec_acc_list, dec_occ_list


# ! Calculate True Positive and False Positive
def TPFP(occ_l= []):
    tp,fp,tn,fn = 0,0,0,0 
    
    # Sum up all occurances of False negatives and positives
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    for i, occ in enumerate(occ_l):
        tp += occ[1]
        tn += occ[0]
        fn += occ[-1]
        fp += occ[2]
    
    # True positive rate - sensitivity 
    tpr = tp / (tp + fn)
    # False Positive - type 1 error
    fpr = fp / (fp + tn)
    
    return tpr, fpr



# ! Run the model
if False:
    filepath = 'Datasets/dec_sep_MPHWAE.csv'
    df = pd.read_csv(filepath)
    df= df.reset_index()
    
    dec_path = 'Datasets/dec_MPHWA.csv'
    dec_df = pd.read_csv(dec_path)
    dec_df = dec_df.reset_index()
    
    X_list = ['ID', 
            'PreviousMedals', 
            'Height_div_avg', 
            'Weight_div_avg', 
            'Age_div_avg'
            ]
    
    Y_list = ['ID', 'MedalEarned']
    
    rng = np.random.default_rng(12345)
    
    RunMore(df, X_list, Y_list, rng, cop = 0.65, times= 50, iterations= 5000, l_rate= 0.02)
    W_array = np.genfromtxt('W.csv', delimiter=',')
    B_array = np.genfromtxt('B.csv', delimiter=',')
    
    Decathlon(dec_df, X_list, Y_list, W_array, B_array, cop= 0.65)
