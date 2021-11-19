import pandas as pd
import addcolumns as add

# ! Get dataset
filepath = 'dec_sep_MPHWA.csv'
df = pd.read_csv(filepath)

vals = ['Height', 'Weight', 'Age']

#listdf = [df['Sport'] for x in df]
#print(listdf)

#df_sliced_dict = {}

#for event in df['Event'].unique():
#    df_sliced_dict[event] = df[df['Event'] == event]

#print(df_sliced_dict)

def div_from_avg_per_year(df, vals):
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


x = add.DeviationAverageEvent(df, vals)

print(x)