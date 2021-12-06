import pandas

'''
# * Test parameters on random decathlon athletes
def DecathlonEven(df, W_list, B_list, times, dec_times=20):
    dec_acc_list = []
    
    #Test parameters on dec
    for i in range(len(W_list)):
        for x in range(dec_times):
            df_1, df_0 = EvenDF(dec_df)
            df_list = [df_1, df_0]
            df = pd.concat(df_list)
            
            # Reduce and split X and Y dataframes
            X_dec = df[X_list]
            Y_dec = df[Y_list]
            
            # Create csv files
            X_dec.to_csv('X_dec.csv', index=False)
            Y_dec.to_csv('Y_dec.csv', index=False)
            
            # Import and reshape dec data
            X_dec, Y_dec = ImportReshape('dec')
            
            dec_acc, dec_occurance_dic = Accuracy(X_dec, Y_dec, W_list[i], B_list[i])
            dec_acc_list.append(dec_acc)
    
    PrintAccReport(dec_acc_list, times, 'Decathlon accuracy')

def ImportReshape(switch):
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
'''