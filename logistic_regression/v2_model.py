import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


# ! Create train and test datasets
def TrainValidate(df, X_list, Y_list):
    # Define features and results
    X = df[X_list]
    Y = df[Y_list]
    
    # define train and validate sets
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size= 0.3, random_state= 44)
    
    # Up-sample data
    over_sampler = RandomOverSampler(random_state=42)
    X_train, Y_train = over_sampler.fit_resample(X_train, Y_train)
    
    return X_train, X_validate, Y_train, Y_validate


# ! Reshapes X and Y files
def Reshape(X, Y):
    # Define dataframes as variables
    X = X.values
    Y = Y.values
    
    # Reshape dataframes to appropriate shape
    X = X.T
    Y = Y.reshape(1, X.shape[1])
    
    return X, Y


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


# * Classify winners and losers
def Classify(X, W, B, cop, bin= True):
    lin_func = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lin_func) # Sigmoid function
    
    # Make sf binary array with data type int64
    sf = sf > cop # Sets sf to one if > 0 or 0 if < 0
    sf = np.array(sf, dtype = 'int64')
    
    return sf


# ! Run model
def RunModel(df, X_list, Y_list, cop, iterations, l_rate):
    X_t, X_val, Y_t, Y_val = TrainValidate(df, X_list, Y_list)
    
    # Import and reshape training and validation dataframes
    X_t, Y_t = Reshape(X_t, Y_t)
    X_val, Y_val = Reshape(X_val, Y_val)
    
    # Call Model function
    W, B, cost_list = Model(X_t, Y_t, l_rate, iterations)
    
    sf_val = Classify(X_val, W, B, cop)
    val_acc, val_occ_dic = Accuracy(sf_val, Y_val)
    
    print(f'Accuracy of the model is : {round(val_acc * 100, 2)}')
    plt.plot(np.arange(iterations), cost_list)
    plt.show()
    
    return W, B, val_acc, val_occ_dic, X_val, Y_val


# ! Run parameters on decathlon athletes
def Decathlon(df, X_list, Y_list, W, B, cop):
    # Reduce and split X and Y dataframes
    X_dec = df[X_list]
    Y_dec = df[Y_list]
    
    X_dec, Y_dec = Reshape(X_dec, Y_dec)
    
    sf = Classify(X_dec, W, B, cop)
    dec_acc, dec_occ = Accuracy(sf, Y_dec)
    
    print(f'Accuracy of the model on decathlon is : {round(dec_acc * 100, 2)}')
    
    return dec_acc, dec_occ


# ! Predict probability
def PredProb(X, W, B):
    lin_func = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lin_func) # Sigmoid function
    
    return sf


# ! Plot ROC-curve
def ROC(X_val, Y_val, W, B):
    pred_prob = PredProb(X_val, W, B)
    false_positive_rate, true_positive_rate, threshold = roc_curve(Y_val.T, pred_prob.T)
    
    plt.subplots(1, figsize=(7,7))
    plt.title('Receiver Operating Characteristic - Logistic regression')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# ! Plot confusion matrix
def Confusion(acc, occ):
    tp,fp,tn,fn = 0,0,0,0
    
    # Sum up all occurances of False negatives and positives
    # 1 = True Pos, 0 = True Neg, -1 = False Neg, 2 = False Pos 
    tp += occ[1]
    tn += occ[0]
    fn += occ[-1]
    fp += occ[2]
    
    cm =    [[tn, fp],
            [fn, tp]]
    
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", square = True)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(round(acc * 100, 2))
    plt.title(all_sample_title)
    plt.show()


# ! Plot normal distribution
def NormDist(X, W, B):
    x = PredProb(X, W, B)
    
    sns.distplot(x.T, fit=norm)
    
    #Now plot the distribution
    plt.ylabel('Frequency')
    plt.title('Probability Prediction Distribution')
    
    #Get also the QQ-plot
    plt.figure()
    stats.probplot(x[0], plot= plt)
    plt.show()


# ! Logistic regression model using sklearn
def LogResModel(df, X_list, Y_list):
    X_t, X_val, Y_t, Y_val = TrainValidate(df, X_list, Y_list)
    
    Y_tr = np.ravel(Y_t)
    
    logisticRegr = LogisticRegression(max_iter= 10000)
    logisticRegr.fit(X_t, Y_tr)
    
    predictions = logisticRegr.predict(X_val)
    cm = metrics.confusion_matrix(Y_val, predictions)
    score = logisticRegr.score(X_val, Y_val)
    r_score = round(score * 100, 2)
    print(f'Accuracy of sklearn model: {r_score}')
    
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", square = True)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(r_score)
    plt.title(all_sample_title)
    plt.show()


if True:
    # ! Import datasets
    filepath = 'Datasets/dec_sep_MPHWA.csv'
    df = pd.read_csv(filepath)
    df= df.reset_index()
    
    dec_path = 'Datasets/dec_MPHWA.csv'
    dec_df = pd.read_csv(dec_path)
    dec_df = dec_df.reset_index()
    
    X_list = ['Height_div_avg', 'Weight_div_avg', 'Age_div_avg']
    Y_list = ['MedalEarned']
    
    cop = 0.50
    W, B, val_acc, val_occ_dic, X_val, Y_val = RunModel(df, X_list, Y_list, cop, iterations= 5000, l_rate= 0.00015)
    dec_acc, dec_occ = Decathlon(dec_df, X_list, Y_list, W, B, cop)
    #NormDist(X_val, W, B)
    #ROC(X_val, Y_val, W, B)
    #Confusion(val_acc, val_occ_dic)
    #Confusion(dec_acc, dec_occ)
    LogResModel(df, X_list, Y_list)
    
    # TODO Make tabel to present accuracy results of the models
    