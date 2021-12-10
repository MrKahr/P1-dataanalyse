import pandas as pd
import numpy as np


# ! Find diviation from average for given variable and add column to df
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


# ! Find diviation from average per year per event for given variable and add column to df
def DeviationAverageEvent(df, vals):
    df = df.reset_index()
    print(df.info())
    for i in range(len(vals)):
        print(1)        
        val = vals[i]
        df_x = pd.DataFrame(index= range(len(df)), columns= [f'{val}_div_avg'])
        print(2)
        
        # Make df of average values for given variable per year
        df_g = df.groupby(['Year', 'Event'])[val].mean()
        print(3)
        df_g_r = df_g.reset_index()
        print(4)
        
        # iterate over given df
        for i, row in df.iterrows():
            df_c = df_g_r[(df_g_r.Year == row.Year) & (df_g_r.Event == row.Event)]
            df_c = df_c.reset_index()
            
            avg_val = df_c.loc[0, val]
            #print(avg_val)
            
            # Calculate diviation from average for given year
            div_avg = round(row[val] - avg_val, 2)
            
            #print(i)
            
            # Add diviation to new column in df
            df_x.at[i, f'{val}_div_avg'] = div_avg
        
        print(5)
        df = pd.concat([df,df_x], axis= 1)
        print(df.info())
    
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


# ! Add previous medals earned to observation
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


# ! Add NOC strength in given sport to observation
def NOCBlackBox(df):
    # Amount of medals won by each NOC
    ol_medal = df.groupby(['NOC'])['MedalEarned'].sum().sort_values(ascending=False)
    
    # Occurance count of each NOC
    ol_count = df['NOC'].value_counts()
    
    ol_tabel = pd.DataFrame({'MedalSum': ol_medal, 'NOCOcc': ol_count}) # Make tabel
    
    # Calculate MedalEarned per athlete by NOC
    ol_tabel['MedPerAth'] = ol_tabel.apply(lambda row: round(row.MedalSum / row.NOCOcc, 2), axis=1)
    ol_tabel = ol_tabel.sort_values(by= 'MedPerAth', ascending= False)
    #print(ol_tabel)
    ol_tabel = ol_tabel.reset_index()
    return ol_tabel


def NOCStrength(df):
    nocYearDict = {}
    
    for i, row in df.iterrows():
        year = row['Year']
        noc = row['NOC']
        
        key = str(year) + noc
        
        if key in nocYearDict:
            # already Registered
            val = nocYearDict[key]
        else:
            dfBefore = df[df.Year > year]
            
            if len(dfBefore) > 0:
                nocDF = NOCBlackBox(dfBefore)
                # Register new keys
                for j, row2 in nocDF.iterrows():
                    tempNoc = row2['index'] # the NOC is the index of the DF
                    tempKey = str(year) + tempNoc
                    
                    nocYearDict[tempKey] = row2['MedPerAth']
                
                if key in nocYearDict:
                    # newly registered and now in the DF
                    val = nocYearDict[key]
                else:
                    # newly registered and still not in the DF
                    nocYearDict[key] = 0
                    val = nocYearDict[key]
            else:
                # Not registered and not in the df
                nocYearDict[key] = 0
                val = nocYearDict[key]
        df.loc[i,'NOC_advantage'] = val
    
    return df