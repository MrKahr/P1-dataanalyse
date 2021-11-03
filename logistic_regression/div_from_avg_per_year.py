import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ! Get dataset
filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

# ! reduce df
df = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year >= 1960) &
        (df.Sport == 'Athletics')
        ]


# ! Find diviation from average for given variable
def div_from_avg_per_year(df, val):
    # df of average values for given variable per year
    df_g = df.groupby(['Year'])[val].mean()
    df_g = df_g.reset_index()
    df_g = df_g.set_index('Year')

    # iterate over given df
    for i, row in df.iterrows():
        # Locate average value for given year
        avg_val = df_g.loc[row['Year'], val]
        # Calculate diviation from average for given year
        div_avg = row[val] - avg_val
        
        # Add diviation to new column in df
        df.at[i, f'{val}_div_avg'] = div_avg
    
    return df


print(div_from_avg_per_year(df, 'Height'))