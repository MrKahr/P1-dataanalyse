import pandas as pd
import Add_columns_to_df as add

# ! Get dataset
filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

#listdf = [df['Sport'] for x in df]
#print(listdf)

df_sliced_dict = {}

for event in df['Event'].unique():
    df_sliced_dict[event] = df[df['Event'] == event]

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