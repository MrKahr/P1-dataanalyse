import pandas as pd
import numpy as np


# ! Find diviation from average for given variable and add column to df
# TODO add so that it can differentiate between events
def DeviationAverage(df, vals):
    for i in range(len(vals)):
        val = vals[i]
        
        # Make df of average values for given variable per year
        df_g = df.groupby(['Year'])[val].mean()
        df_g_r = df_g.reset_index()
        df_g_s = df_g_r.set_index('Year')

        # iterate over given df
        for i, row in df.iterrows():
            # Locate average value for given year
            avg_val = df_g_s.loc[row['Year'], val]
            # Calculate diviation from average for given year
            div_avg = round(row[val] - avg_val, 2)
            
            # Add diviation to new column in df
            df.at[i, f'{val}_div_avg'] = div_avg
    
    return df


# ! Add MedalEarned to df
def MedalEarned(df):
    # Locate and define medals
    conditions = [(df['Medal'] == 'Gold'),
                  (df['Medal'] == 'Silver'),
                  (df['Medal'] == 'Bronze'),
                  (df['Medal'] == 'NA')]

    # Define values for medals
    values = [1,1,1,0]

    # Add MedalEarned column with defined value to observations
    df['MedalEarned'] = np.select(conditions,values)
    
    return df


# ! add previous medals earned to observation
def PreviousMedals(df):
    # * Create empty dict
    ID_medals = {}
    
    # * Create PreviousMedals df
    df_prev_med = pd.DataFrame(index= range(len(df)), columns= ['PreviousMedals'])
    
    # * sort by ID then by year
    df = df.sort_values(by= 'ID', ascending= True)
    df = df.sort_values(by= 'Year', ascending= True)

    # * iterate through sorted df and add previous medals to athlete
    for i, row in df.iterrows():
        if row['MedalEarned'] == 1:
            if row['ID'] in ID_medals:
                df_prev_med.at[i, 'PreviousMedals'] = ID_medals[row['ID']]
                
                ID_medals[row['ID']] += 1
            
            else:
                ID_medals[row['ID']] = 1
                
                df_prev_med.at[i, 'PreviousMedals'] = 0
        
        else: 
            if row['ID'] in ID_medals:
                df_prev_med.at[i, 'PreviousMedals'] = ID_medals[row['ID']]
            
            else:
                df_prev_med.at[i, 'PreviousMedals'] = 0
    
    # * Add previous medal column to df
    df = pd.concat([df,df_prev_med], axis= 1)
    
    if False:
        list = [ID for ID, occurrences in ID_medals.items() if occurrences >= 3]
        print(list)

    return df
