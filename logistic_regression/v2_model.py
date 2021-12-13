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
    
    print(f'Accuracy of the model on decathlon is : {dec_acc}')
    
    return dec_acc, dec_cm


# ! Predict probability
def PredProb(X, W, B):
    lf = np.dot(W.T, X) + B # Linear function
    sf = Sigmoid(lf) # Sigmoid function
    
    return sf


# ! Plot ROC-curve
def ROC(X_val, Y_val, W, B):
    pred_prob = PredProb(X_val, W, B) # Get predicted probabilities
    false_positive_rate, true_positive_rate, threshold = roc_curve(Y_val.T, pred_prob.T) # Calculate ROC-curve features
    
    # Plot ROC-curve
    plt.subplots(1, figsize=(7,7))
    plt.title('Receiver Operating Characteristic - Logistic regression')
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


# ! Plot normal distribution
def NormDist(X, W, B):
    x = PredProb(X, W, B) # Get predicted probability
    sns.distplot(x.T, fit=norm) # Define distributions
    
    # Plot distributions
    plt.ylabel('Frequency')
    plt.title('Probability Prediction Distribution')
    
    # QQ-plot
    plt.figure()
    stats.probplot(x[0], plot= plt)
    plt.show()


# ! Logistic regression model using sklearn
def SklearnModel(df, X_list, Y_list):
    X_t, X_val, Y_t, Y_val = TrainValidate(df, X_list, Y_list)
    
    Y_tr = np.ravel(Y_t) # Change to shape (n, )
    
    logisticRegr = LogisticRegression() # Define the model
    logisticRegr.fit(X_t, Y_tr) # Train model
    
    predictions = logisticRegr.predict(X_val) # Make predictions
    
    # Make cm and calculate acc
    cm = metrics.confusion_matrix(Y_val, predictions)
    score = logisticRegr.score(X_val, Y_val)
    sk_acc = f'{round(score * 100, 2)} %'
    
    print(f'Accuracy of the sklearn model is: {sk_acc}')
    
    return sk_acc, cm


# ! Result tabel
def PrintModelResults(acc_column, cm_list):
    tpr_column = []
    fpr_column = []
    
    # Calcluate the True Positive Rate and False Positive Rate
    for i, cm in enumerate(cm_list):
        tpr = f'{round(cm[1][1] / (cm[1][1] + cm[1][0]), 2)}'
        fpr = f'{round(cm[0][1] / (cm[0][1] + cm[0][0]), 2)}'
        tpr_column.append(tpr)
        fpr_column.append(fpr)
    
    # Create dataframe of acc, tpr and fpr columns
    report = pd.DataFrame({
                        'ACC': acc_column,
                        'TPR': tpr_column,
                        'FPR': fpr_column
                        },
                        index= ['Validate', 'Decathlon', 'Sklearn'])
    
    # Plot dataframe as table
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    t= ax.table(cellText=report[['ACC', 'TPR', 'FPR']].head( n=3).values, # Set header names
                colWidths = [0.2]*len(report.columns), colColours = ['royalblue']*3, # Set header cell size and colour
                rowLabels=report.index ,colLabels=report.columns,  loc='center') # Fill tabel
    
    # Set layout features
    t.auto_set_font_size(False) 
    t.set_fontsize(8)
    fig.tight_layout()
    
    # Colour content cells white
    for i in range(3):
        cell = t[0,i]
        cell.get_text().set_color('white')
        
    # Set header cells font to bold
    for (row, col), cell in t.get_celld().items():
        if (row == 0) or (col == 5):
            cell.set_text_props(fontproperties=FontProperties(weight = 'bold'))
    
    plt.show()


if True:
    # ! Import datasets
    filepath = 'Datasets/expert_data.csv'
    df = pd.read_csv(filepath)
    df= df.reset_index()
    
    dec_path = 'Datasets/decathlon_data.csv'
    dec_df = pd.read_csv(dec_path)
    dec_df = dec_df.reset_index()
    
    # ! Training features
    X_list = ['PreviousMedals', 'NOC_advantage', 'Height_div_avg', 'Weight_div_avg', 'Age_div_avg']
    Y_list = ['MedalEarned']
    
    # ! Models an tests
    cop = 0.6
    W, B, val_acc, val_cm, X_val, Y_val = RunModel(df, X_list, Y_list, cop, iterations= 80000, learning_rate= 0.0223)
    dec_acc, dec_cm = Decathlon(dec_df, X_list, Y_list, W, B, cop)
    sk_acc, sk_cm = SklearnModel(df, X_list, Y_list)
    
    acc_list = [val_acc, dec_acc, sk_acc]
    cm_list = [val_cm, dec_cm, sk_cm]
    
    # ! Result visualisations
    NormDist(X_val, W, B)
    ROC(X_val, Y_val, W, B)
    PrintModelResults(acc_list, cm_list)
    Confusion(val_acc, val_cm, 'Validation Matrix')
    Confusion(dec_acc, dec_cm, 'Decathlon Matrix')
    Confusion(sk_acc, sk_cm, 'Sklearn Matrix')
