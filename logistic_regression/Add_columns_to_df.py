import pandas as pd
import numpy as np
# TODO add so that it can funktion for different events


# ! Find diviation from average for given variable and add column to df
def div_from_avg_per_year(df, vals):
    # df of average values for given variable per year
    for i in range(len(vals)):
        val = vals[i]
        
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
    conditions = [(df['Medal'] == 'Gold'),
                (df['Medal'] == 'Silver'),
                (df['Medal'] == 'Bronze'),
                (df['Medal'] == 'NA')]

    values = [1,1,1,0]

    df['MedalEarned'] = np.select(conditions,values)
    
    return df
#vals = ['Height', 'Weight', 'Age']
#Events = ['Athletics Men\'s 100 metres', 'Athletics Men\'s 400 metres']