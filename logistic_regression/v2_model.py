'''
Credit to GitHub user Jaimin09
Link: https://github.com/Jaimin09/Coding-Lane-Assets/tree/main/Logistic%20Regression%20in%20Python%20from%20Scratch
Last accessed: 28/10/2021
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
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
    
    cm = [[tn, fp],
        [fn, tp]]
    
    # Calculate accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc, cm


# * Classify winners and losers
def Classify(X, W, B, cop):
    lf = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lf) # Sigmoid function
    
    # Make sf binary array with data type int64
    sf = sf > cop # Sets sf to one if > 0 or 0 if < 0
    sf = np.array(sf, dtype = 'int64')
    
    return sf


# ! Run model
def RunModel(df, X_list, Y_list, cop, iterations, learning_rate):
    X_train, X_validate, Y_train, Y_validate = TrainValidate(df, X_list, Y_list)
    
    # Import and reshape training and validation dataframes
    X_train, Y_train = Reshape(X_train, Y_train)
    X_validate, Y_validate = Reshape(X_validate, Y_validate)
    
    # Call Model function
    W, B, cost_list = Model(X_train, Y_train, learning_rate, iterations)
    
    # Calculate accuracy of model
    val_sf = Classify(X_validate, W, B, cop)
    val_acc, val_cm = Accuracy(val_sf, Y_validate)
    val_acc = f'{round(val_acc * 100, 2)} %'
    
    # Plot cost value over model iterations and print accuracy
    print(f'Accuracy of the model is : {val_acc}')
    plt.plot(np.arange(iterations), cost_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    return W, B, val_acc, val_cm, X_validate, Y_validate


# ! Run parameters on decathlon athletes
def Decathlon(df, X_list, Y_list, W, B, cop):
    # Reduce and split X and Y dataframes
    X_dec = df[X_list]
    Y_dec = df[Y_list]
    
    X_dec, Y_dec = Reshape(X_dec, Y_dec)
    
    # Calculate and print accuracy of model on decathlon athletes
    sf = Classify(X_dec, W, B, cop)
    dec_acc, dec_cm = Accuracy(sf, Y_dec)
    dec_acc = f'{round(dec_acc * 100, 2)} %'
    
    print(f'Accuracy of the model on decathlon is: {dec_acc}')
    
    return dec_acc, dec_cm, X_dec, Y_dec


# ! Random predictions
def RandomPredictions(X, Y):
    np.random.seed(1)
    prediction = np.random.randint(2, size= len(X.T))
    rand_acc, rand_cm = Accuracy(prediction, Y)
    rand_acc = f'{round(rand_acc * 100, 2)} %'
    
    print(f'Accuracy of random predictions is: {rand_acc}')
    
    return rand_acc, rand_cm


# ! Predict probability
def PredProb(X, W, B):
    lf = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lf) # Sigmoid function
    
    return sf


# ! Plot ROC-curve
def ROC(X_val, Y_val, W, B):
    pred_prob = PredProb(X_val, W, B) # Get predicted probabilities
    false_positive_rate, true_positive_rate, threshold = roc_curve(Y_val.T, pred_prob.T) # Calculate ROC-curve features
    
    #Locate cop off point (threshold) with highest TPR and lowest FPR
    i = np.arange(len(true_positive_rate)) 
    roc = pd.DataFrame({'tf' : pd.Series(true_positive_rate - (1 - false_positive_rate), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
    opt_cop = round(list(roc_t['threshold'])[0], 2)
    
    # Plot ROC-curve
    plt.subplots(1, figsize=(7,7))
    plt.title(f'Receiver Operating Characteristic \nOptimal cut off point: {opt_cop}')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# ! Plot confusion matrix
def Confusion(acc, cm, data_title= ''):
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", square = True)
    plt.ylabel('Actual outcome')
    plt.xlabel('Predicted outcome')
    plt.title(f'{data_title} {acc}')
    plt.show()


# ! Logistic regression model using sklearn
def SklearnModel(df, dec_df, X_list, Y_list, X_list_dec):
    X_t, X_val, Y_t, Y_val = TrainValidate(df, X_list, Y_list)
    
    Y_tr = np.ravel(Y_t) # Change to shape (n, )
    
    logisticRegr = LogisticRegression() # Define the model
    logisticRegr.fit(X_t, Y_tr) # Train model
    
    predictions = logisticRegr.predict(X_val) # Make predictions
    
    # Make cm and calculate acc
    cm = metrics.confusion_matrix(Y_val, predictions)
    score = logisticRegr.score(X_val, Y_val)
    sk_acc = f'{round(score * 100, 2)} %'
    
    print(f'Accuracy of the sklearn val model is: {sk_acc}')
    
    # Reduce and split X and Y dataframes
    X_dec = dec_df[X_list_dec]
    Y_dec = dec_df[Y_list]
    
    X_dec, Y_dec = Reshape(X_dec, Y_dec)
    
    predictions_dec = logisticRegr.predict(X_dec.T) # Make predictions
    
    # Make cm and calculate acc
    cm_dec = metrics.confusion_matrix(Y_dec.T, predictions_dec)
    score_dec = logisticRegr.score(X_dec.T, Y_dec.T)
    sk_acc_dec = f'{round(score_dec * 100, 2)} %'
    
    print(f'Accuracy of the sklearn dec model is: {sk_acc_dec}')
    
    return sk_acc, cm, sk_acc_dec, cm_dec


# ! Result tabel
def PrintModelResults(acc_column, cm_list, name_list):
    tpr_column = []
    fpr_column = []
    fdr_column = []
    ppv_column = []
    
    # Calcluate the True Positive Rate, False Positive Rate and False Discovery Rate
    for i, cm in enumerate(cm_list):
        tpr = f'{round(cm[1][1] / (cm[1][1] + cm[1][0]), 2)}'
        fpr = f'{round(cm[0][1] / (cm[0][1] + cm[0][0]), 2)}'
        fdr = f'{round(cm[0][1] / (cm[0][1] + cm[1][1]), 2)}'
        ppv = f'{round(cm[1][1] / (cm[0][1] + cm[1][1]), 2)}'
        tpr_column.append(tpr)
        fpr_column.append(fpr)
        fdr_column.append(fdr)
        ppv_column.append(ppv)
    
    # Create dataframe of acc, tpr, fpr and fdr columns
    report = pd.DataFrame({
                        'ACC': acc_column,
                        'TPR': tpr_column,
                        'FPR': fpr_column,
                        'FDR': fdr_column,
                        'PPV': ppv_column
                        },
                        index= name_list)
    
    column_count = len(report.columns)
    
    # Plot dataframe as table
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    t= ax.table(cellText=report[['ACC', 'TPR', 'FPR', 'FDR', 'PPV']].head( n= column_count).values, # Set header names
                colWidths = [0.2]*len(report.columns), colColours = ['royalblue']*column_count, # Set header cell size and colour
                rowLabels=report.index ,colLabels=report.columns,  loc='center') # Fill tabel
    
    # Set layout features
    t.auto_set_font_size(False) 
    t.set_fontsize(8)
    fig.tight_layout()
    
    # Colour content cells white
    for i in range(column_count):
        cell = t[0,i]
        cell.get_text().set_color('white')
    
    # Set header cells font to bold
    for (row, col), cell in t.get_celld().items():
        if (row == 0) or (col == column_count):
            cell.set_text_props(fontproperties=FontProperties(weight = 'bold'))
    
    plt.show()


if __name__ == '__main__':
    # ! Import datasets
    filepath = 'Datasets/expert_data.csv'
    df = pd.read_csv(filepath)
    df= df.reset_index()
    
    dec_path = 'Datasets/decathlon_data.csv'
    dec_df = pd.read_csv(dec_path)
    dec_df = dec_df.reset_index()
    
    # ! Training features
    X_list = ['PreviousMedals', 'NOC_advantage', 'Height_Dev_Event', 'Weight_Dev_Event', 'Age_Dev_Event']
    X_list_dec = ['PreviousMedals', 'NOC_advantage', 'Height_Dev', 'Weight_Dev', 'Age_Dev']
    Y_list = ['MedalEarned']
    
    # ! Models an tests
    # * Normal deviation
    #cop = 0.4
    #W, B, val_acc, val_cm, X_val, Y_val = RunModel(df, X_list_dec, Y_list, cop, iterations= 80000, learning_rate= 0.0223)
    
    # * Deviation per Event
    cop = 0.42
    W, B, val_acc, val_cm, X_val, Y_val = RunModel(df, X_list, Y_list, cop, iterations= 19000, learning_rate= 0.13)
    
    dec_acc, dec_cm, X_dec, Y_dec = Decathlon(dec_df, X_list_dec, Y_list, W, B, cop)
    sk_acc_val, sk_cm_val, sk_acc_dec, sk_cm_dec = SklearnModel(df, dec_df, X_list, Y_list, X_list_dec)
    rand_acc, rand_cm = RandomPredictions(X_dec, Y_dec)
    
    acc_list = [val_acc, dec_acc, rand_acc]
    cm_list = [val_cm, dec_cm, rand_cm]
    name_list = ['Validate', 'Decathlon', 'Random']
    
    acc_list_sk = [sk_acc_val, sk_acc_dec]
    cm_list_sk = [sk_cm_val, sk_cm_dec]
    name_list_sk = ['Sklearn_val', 'Sklearn_dec']
    
    # ! Result visualisations
    #ROC(X_val, Y_val, W, B)
    PrintModelResults(acc_list, cm_list, name_list)
    PrintModelResults(acc_list_sk, cm_list_sk, name_list_sk)
    Confusion(val_acc, val_cm, 'Validation Matrix')
    Confusion(dec_acc, dec_cm, 'Decathlon Matrix')
    
    Confusion(sk_acc_val, sk_cm_val, 'Sklearn val Matrix')
    Confusion(sk_acc_dec, sk_cm_dec, 'Sklearn dec Matrix')
    
    Confusion(rand_acc, rand_cm, 'Random Pred. Matrix')
