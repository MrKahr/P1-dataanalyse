'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


# ! Functions that manipulate dataframes and csv files
# * Reshapes X and Y files
def Reshape(X, Y):
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
        lf = np.dot(W.T, X) + B # Linear function
        sf = Sigmoid(lf) # Sigmoid function
        
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
    
    val_sf = Classify(X_validate, W, B, cop)
    val_acc, val_occ = Accuracy(val_sf, Y_validate)
    
    return W, B, val_acc, val_occ


# * Classify winners and losers
def Classify(X, W, B, cop):
    lf = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lf) # Sigmoid function
    
    # Make sf binary array with data type int64
    sf = sf > cop # Sets sf to one if > 0 or 0 if < 0
    sf = np.array(sf, dtype = 'int64')
    
    return sf


# * Calculate accuracy of the model
def Accuracy(sf, Y):
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    predictions = sf * 2 - Y
    occurance = [[x, list(predictions[0]).count(x)] for x in set(list(predictions[0]))]
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
def PrintAccReport(acc_lists, occ_lists):
    avg_acc_column = []
    min_acc_column = []
    max_acc_column = []
    tpr_column = []
    fpr_column = []
    fdr_column = []
    ppv_column = []
    
    for i, acc_list in enumerate(acc_lists):
        # Calculate average, min and max accuracy
        acc_avg = round(sum(acc_list) / len(acc_list)*100, 2)
        acc_min = round(min(acc_list)*100, 2)
        acc_max = round(max(acc_list)*100, 2)
        
        avg_acc_column.append(f'{acc_avg} %')
        min_acc_column.append(f'{acc_min} %')
        max_acc_column.append(f'{acc_max} %')
        
        #Calcluate the True Positive Rate and False Positive Rate
    for i, occ_list in enumerate(occ_lists):
        tpr,fpr,fdr,ppv = TPFP(occ_list)
        tpr_column.append(format(tpr, ".2f"))
        fpr_column.append(format(fpr, ".2f"))
        fdr_column.append(format(fdr, ".2f"))
        ppv_column.append(format(ppv, ".2f"))
    
    report = pd.DataFrame({
                        'Avg. Acc.' : avg_acc_column,
                        'Min. Acc.': min_acc_column,
                        'Max. Acc.': max_acc_column,
                        'TPR': tpr_column,
                        'FPR': fpr_column,
                        'FDR': fdr_column,
                        'PPV': ppv_column
                        },
                        index= ['Validate', 'Test', 'Decathlon'])
    
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    t= ax.table(cellText=report[['Avg. Acc.', 'Min. Acc.', 'Max. Acc.', 'TPR', 'FPR', 'FDR', 'PPV']].head( n=7).values,
                colColours = ['royalblue']*7,
                rowLabels=report.index ,colLabels=report.columns,  loc='center')
    
    t.auto_set_font_size(False) 
    t.set_fontsize(8)
    fig.tight_layout()
    
    for i in range(7):
        cell = t[0,i]
        cell.get_text().set_color('white')
    
    for (row, col), cell in t.get_celld().items():
        if (row == 0) or (col == 7):
            cell.set_text_props(fontproperties=FontProperties(weight = 'bold'))
    
    plt.show()


# * Run multiple iterations of the model
def RunMore(df, X_list, Y_list, rng, cop, times, iterations, l_rate, save_par= False):
    W_list = []
    B_list = []
    val_acc_list = []
    val_occ_list = []
    
    # Create test sample
    df_testless, X_test, Y_test = TestSampler(df, rng, X_list, Y_list)
    
    for i in range(times):
        # Run model
        W, B, val_acc, val_occ = RunModel(df_testless, rng, cop, iterations, l_rate, X_list, Y_list)
        
        # Append parameters, accuracy and occurances to lists
        W_list.append(W)
        B_list.append(B)
        val_acc_list.append(val_acc)
        val_occ_list.append(val_occ)
        
        # Progress bar
        if len(W_list) % 10 == 0:
            print(f'{times - len(W_list)} runs left.')
    
    X_test, Y_test = Reshape(X_test, Y_test)
    test_occ_list = []
    test_acc_list = []
    
    # Test parameters on test data
    for i in range(len(W_list)):
        test_sf = Classify(X_test, W_list[i], B_list[i], cop)
        test_acc, test_occ = Accuracy(test_sf, Y_test)
        test_acc_list.append(test_acc)
        test_occ_list.append(test_occ)
    
    W_array = np.concatenate(W_list, axis=1)
    B_array = np.stack(B_list)
    
    if save_par:
        np.savetxt('W.csv', W_array, delimiter= ',')
        np.savetxt('B.csv', B_array, delimiter= ',')
    
    return val_acc_list, test_acc_list, W_array, B_array, val_occ_list, test_occ_list


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
        W_par = np.array([W_array[0][i], W_array[1][i], W_array[2][i]], ndmin= 0)
        dec_sf = Classify(X_dec, W_par, B_array[i], cop)
        dec_acc, dec_occ = Accuracy(dec_sf, Y_dec)
        dec_acc_list.append(dec_acc)
        dec_occ_list.append(dec_occ)
    
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
    # False Discovery Rate
    fdr = fp / (tp + fp)
    # Positive Predictive Value
    ppv = tp / (tp + fp)
    
    return tpr, fpr, fdr, ppv

# ! Plot confusion matrix
def Confusion(acc, occ, times = 50, data_title = ''):
    tp,fp,tn,fn = 0,0,0,0
    
    # Sum up all occurances of False negatives and positives
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    for i in range(times):
        tp += occ[i][1]
        tn += occ[i][0]
        fn += occ[i][-1]
        fp += occ[i][2]
    
    # True positive rate - sensitivity 
    tpr = tp / (tp + fn)
    # False Positive - type 1 error
    fpr = fp / (fp + tn)
    
    print(f'True positive rate: {round(tpr*100, 2)}')
    print(f'False positive rate: {round(fpr*100, 2)}')
    
    cm = [[tn, fp],
        [fn, tp]]
    
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", square = True)
    plt.ylabel('Actual outcome')
    plt.xlabel('Predicted outcome')
    plt.title(data_title)
    plt.show()


# ! Run the model
if __name__ == '__main__':
    filepath = 'Datasets/Datasets_we_dont_need/dec_sep_MPHWA.csv'
    df = pd.read_csv(filepath)
    df= df.reset_index()
    
    dec_path = 'Datasets/Datasets_we_dont_need/dec_MPHWA.csv'
    dec_df = pd.read_csv(dec_path)
    dec_df = dec_df.reset_index()
    
    X_list = ['Height', 'Weight', 'Age']
    Y_list = ['MedalEarned']
    
    rng = np.random.default_rng(12345)
    
    val_acc_list, test_acc_list, W_array, B_array, val_occ_list, test_occ_dic_list = RunMore(df, X_list, Y_list, rng, cop = 0.50, times= 50, iterations= 5000, l_rate= 0.00015)
    
    dec_acc_list, dec_occ_list = Decathlon(dec_df, X_list, Y_list, W_array, B_array, cop= 0.50)
    
    list_of_acc_lists = [val_acc_list, test_acc_list, dec_acc_list]
    PrintAccReport([val_acc_list, test_acc_list, dec_acc_list], [val_occ_list,test_occ_dic_list,dec_occ_list])
    
    
    Confusion(sum(val_acc_list)/len(val_acc_list),val_occ_list, data_title = 'Validation Matrix')
    Confusion(sum(test_acc_list)/len(test_acc_list),test_occ_dic_list, data_title = 'Test Matrix')
    Confusion(sum(dec_acc_list)/len(dec_acc_list),dec_occ_list, data_title = 'Decathlon Matrix')
