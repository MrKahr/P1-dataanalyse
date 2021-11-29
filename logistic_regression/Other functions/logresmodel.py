'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ! Set seed and seed calling function
rng = np.random.default_rng(12345)

# ! Get dataset
filepath = 'dec_sep_MPHWAE.csv'
df = pd.read_csv(filepath)
df= df.reset_index()

dec_path = 'dec_MPHWA.csv'
dec_df = pd.read_csv(dec_path)
dec_df = dec_df.reset_index()

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
def EvenDF(df):
    # Split dataframe into won a medal and didnt win a medal
    df_1 = df[df.MedalEarned == 1]
    df_0 = df[df.MedalEarned == 0]
    
    # Randomly sample df_0 to size of df_1
    df_0 = df_0.sample(n = len(df_1), random_state=rng.integers(1000))
    
    return df_1, df_0


# * Make df_test (X_test and Y_test)
def TestSampler(df, X_list, Y_list):
    # Split dataframe into won a medal and didnt win a medal
    df_1 = df[df.MedalEarned == 1]
    df_0 = df[df.MedalEarned == 0]
    
    # Randomly sample test df_1 and df_0
    df_1_test = df_1.sample(n = 100, random_state=rng.integers(1000))
    df_0_test = df_0.sample(n = 100, random_state=rng.integers(1000))
    
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
def TrainValidate(df, X_list, Y_list):
    # Randomly sample df_0 to size of df_1
    df_1, df_0 = EvenDF(df)
    
    # Randomly sample validate df_1 and df_0
    df_1_validate = df_1.sample(frac= 0.2, random_state=rng.integers(1000))
    df_0_validate = df_0.sample(frac= 0.2, random_state=rng.integers(1000))
    
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
# * Test X and Y shapes (prints to varify)
def Test(X_train, Y_train, X_validate, Y_validate):
    print("Shape of X_train : ", X_train.shape)
    print("Shape of Y_train : ", Y_train.shape)
    print("Shape of X_test : ", X_validate.shape)
    print("Shape of Y_test : ", Y_validate.shape)
    print('')


# * Sigmoid function
def Sigmoid(x):
    return 1/(1 + np.exp(-x))


# * The model
def Model(X, Y, learning_rate, iterations, cost_progress= False):
    m = X.shape[1] # Observations
    n = X.shape[0] # Types of parameters
    
    W = np.zeros((n,1)) # All a parameters
    B = 0
    
    cost_list = [] # Empty cost list
    
    for i in range(iterations):
        lin_func = np.dot(W.T, X) + B # Linear function
        sig_func = Sigmoid(lin_func) # Sigmoid function
        
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


def AltClassify(df, X, Y, W, B, wg):
    # Calculate predictions
    lf = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lf) # Sigmoid function
    sf_T = sf.T
    
    # Add predictions to df
    sf_pd = pd.DataFrame(sf_T, columns= ['Prediction'])
    df_sf = pd.concat([df, sf_pd], axis= 1)
    
    # Find highest predictions for each year
    df1 = df_sf.groupby(['Year']).apply(lambda x: x.sort_values(['Prediction'], ascending= False))
    df2 = df1.reset_index(drop= True)
    df3 = df2.groupby('Year').head(wg)
    index_list = df3['index'].tolist()
    
    # Set highest predictions to 1 and rest to 0
    for i in range(len(df_sf)):
        if i in index_list:
            df_sf.at[i, 'Prediction'] = 1
        else:
            df_sf.at[i, 'Prediction'] = 0
    
    sf_T_new = df_sf['Prediction'].to_numpy()
    sf_int = np.array(sf_T_new, dtype = 'int64')
    
    # Calculate accuracy
    acc = (1 - np.sum(np.absolute(sf_int - Y)) / Y.shape[1]) * 100
    
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    guesses = sf_int.T * 2 - Y
    occurance = [[x, list(guesses[0]).count(x)] for x in set(list(guesses[0]))]
    occurance_dic = {}
    
    for i in occurance:
        # Assign value to keys e.g. TP = 22 
        occurance_dic[i[0]] = i[1]
    
    return acc, occurance_dic


# * Accuracy test
def Accuracy(X, Y, W, B):
    lin_func = np.dot(W.T, X) + B # Linear function
    sig_func = Sigmoid(lin_func) # Sigmoid function
    
    sig_func = sig_func > 0.6 # Sets sig_func to one if > 0 or 0 if < 0
    
    # Make sig_func array with data type int64
    sig_func = np.array(sig_func, dtype = 'int64') 
    
    # Calculate accuracy
    acc = (1 - np.sum(np.absolute(sig_func - Y)) / Y.shape[1]) * 100
    
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    guesses = sig_func * 2 - Y
    occurance = [[x, list(guesses[0]).count(x)] for x in set(list(guesses[0]))]
    occurance_dic = {1:0, 0:0, -1:0, 2:0}
    
    for i in occurance:
        # Assign value to keys e.g. TP = 22 
        occurance_dic[i[0]] = i[1]
    
    return acc, occurance_dic


# ! The functions that run the model and report on the model
# * Run model
def RunModel(df_testless, iterations, learning_rate, plot_print= False, cost_progress= False, test=False):
    # Make X_train, Y_train, X_validate, Y_validate
    X_train, Y_train, X_validate, Y_validate = TrainValidate(df_testless, X_list, Y_list)
    
    # Import and reshape training and validation dataframes
    X_train, Y_train = Reshape(X_train, Y_train)
    X_validate, Y_validate = Reshape(X_validate, Y_validate)
    
    #Test dataframes
    if test:
        Test(X_train, Y_train, X_validate, Y_validate)
    
    # Call Model function
    W, B, cost_list = Model(X_train, Y_train, learning_rate, iterations, cost_progress)
    
    # Call Accuracy function
    acc, occurance_dic = Accuracy(X_validate, Y_validate, W, B)
    
    # Print accuracy and plot cost value over iterations
    if plot_print:
        print("Accuracy of the model is : ", round(acc, 2), "%")
        plt.plot(np.arange(iterations), cost_list)
        plt.show()
    
    return W, B, acc, occurance_dic


# * Print accuracy
def PrintAccReport(list_of_acc, name):
    # Calculate average, min and max accuracy
    acc_avg = sum(list_of_acc) / len(list_of_acc)
    acc_min = min(list_of_acc)
    acc_max = max(list_of_acc)
    
    # Print average, min and max accuracy
    print(f'{name} results: avg {round(acc_avg, 2)}% ', 
                    f'min {round(acc_min, 2)}% ', 
                    f'max {round(acc_max, 2)}% ')


def round_prcnt(x):
    return f'{round((x * 100), 2)}%'


# * Print false negative reprot
def PredRate(occurance_dic_list, name):
    tp,fp,tn,fn = 0,0,0,0 
    
    # Sum up all occurances of False negatives and positives
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    for i, occ in enumerate(occurance_dic_list):
        tp += occ[1]
        tn += occ[0]
        fn += occ[-1]
        fp += occ[2]
    
    # False Positive - type 1 error
    fpr = fp / (fp + tn) 
    # False negative rate - type 2 error 
    fnr = fn / (tp + fn)
    # True negative rate - specificity 
    tnr = tn / (tn + fp) 
    # False discovery rate 
    fdr = fp / (tp + fp) 
    # True positive rate - sensitivity 
    tpr = tp / (tp + fn) 
    # Positive predictive value - precision 
    ppv = tp / (tp + fp) 
    # Accuracy 
    acc = (tp + tn) / (tp + fp + fn + tn) 
    
    print('TPR: ', round_prcnt(tpr))
    print('FPR: ', round_prcnt(fpr))
    print('TNR: ', round_prcnt(tnr))
    print('acc: ', round_prcnt(acc))


# * Run multiple iterations of the model
def RunMore(times, iterations, learning_rate, plot_print= False, test= False):
    W_list = []
    B_list = []
    acc_list = []
    test_acc_list = []
    occ_dic_list = []
    
    # Create test sample
    df_testless, X_test, Y_test = TestSampler(df, X_list, Y_list)
    
    for i in range(times):
        # Run model
        W, B, acc, occurance_dic = RunModel(df_testless, iterations, learning_rate, plot_print, test)
        
        # Append parameters, accuracy and occurances to lists
        W_list.append(W)
        B_list.append(B)
        acc_list.append(acc)
        occ_dic_list.append(occurance_dic)
        
        # Progress bar
        #if len(acc_list) % 5 == 0:
        #    print(f'{times - len(acc_list)} runs left.')
    
    # Import and reshape test data
    X_test, Y_test = Reshape(X_test, Y_test)
    test_occ_dic_list = []
    
    # Test parameters on test data
    for i in range(len(W_list)):
        test_acc, test_occ_dic = Accuracy(X_test, Y_test, W_list[i], B_list[i])
        test_acc_list.append(test_acc)
        test_occ_dic_list.append(test_occ_dic)
    
    # Print accuracy reports and false negative reports
    #PredRate(occ_dic_list, 'Validate')
    PrintAccReport(acc_list, 'Validate')
    
    #PredRate(test_occ_dic_list, 'Test')
    PrintAccReport(test_acc_list, 'Test')
    print('')
    
    return W_list, B_list


# * Test parameters on decathlon athletes
def Decathlon(df, W_list, B_list):
    wg = 1
    dec_acc_list = []
    dec_occ_list = []
    #dec_alt_acc_list = []
    #dec_alt_occ_list = []
    
    # Reduce and split X and Y dataframes
    X_dec = df[X_list]
    Y_dec = df[Y_list]
    
    # Import and reshape dec data
    X_dec, Y_dec = Reshape(X_dec, Y_dec)
    
    # Test parameters on dec
    for i in range(len(W_list)):
        #dec_alt_acc, dec_alt_occ_dic = AltClassify(dec_df, X_dec, Y_dec, W_list[i], B_list[i], wg)
        dec_acc, dec_occ_dic = Accuracy(X_dec, Y_dec, W_list[i], B_list[i])
        
        #dec_alt_acc_list.append(dec_alt_acc)
        #dec_alt_occ_list.append(dec_alt_occ_dic)
        dec_acc_list.append(dec_acc)
        dec_occ_list.append(dec_occ_dic)
    
    # Print reports
    PredRate(dec_occ_list, 'Decathlon')
    PrintAccReport(dec_acc_list, 'Decathlon')
    
    #PredRate(dec_alt_occ_list, 'Decathlon alt')
    #PrintAccReport(dec_alt_acc_list, 'Decathlon alt')


# ! Variable list for X and Y
X_list = ['ID', 
        'Height_div_avg', 
        'Weight_div_avg', 
        'Age_div_avg'
        ]

#        'PreviousMedals', 

Y_list = ['ID', 'MedalEarned']

#RunModel(df, iterations= 5000, learning_rate= 0.02, plot_print= True, cost_progress= True)
W_list, B_list = RunMore(times = 50, iterations= 5000, learning_rate= 0.02)
Decathlon(dec_df, W_list, B_list)