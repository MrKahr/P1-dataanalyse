import pandas as pd
import numpy as np


# ! Find diviation from average for given variable and add column to df
# TODO add so that it can differentiate between events
def div_from_avg_per_year(df, vals):
    for i in range(len(vals)):
        val = vals[i]
        
        # Make df of average values for given variable per year
        df_g = df.groupby(['Year'])[val].mean()
        df_g = df_g.reset_index()
        df_g = df_g.set_index('Year')

        # iterate over given df
        for i, row in df.iterrows():
            # Locate average value for given year
            avg_val = df_g.loc[row['Year'], val]
            # Calculate diviation from average for given year
            div_avg = round(row[val] - avg_val, 2)
            
            # Add diviation to new column in df
            df.at[i, f'{val}_div_avg'] = div_avg
        
    return df


# ! Add MedalEarned to df
def medal_earned(df):
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
# TODO This is a work in progress
def previous_medals(df):
    # * Create empty dict
    ID_medals = {}
    
    # * Create PreviousMedals df
    df_prev_med = pd.DataFrame(size= (len(df), 1), columns= ['PreviousMedals'])
    
    # * sort by ID then by year
    df.sort_values(by=['MedalEarned'], ascending = False)
    df.sort_values(by=['ID'], ascending = True)
    df.sort_values(by=['Year'], ascending = True)

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
    
    df_and_prev_med = df.join(df_prev_med)
    print(ID_medals)
                

    return df_and_prev_med
