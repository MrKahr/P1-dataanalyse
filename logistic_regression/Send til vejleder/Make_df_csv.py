import pandas as pd
import Add_columns_to_df as add

# ! Get dataset and variables
filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

vals = ['Height', 'Weight', 'Age']

# ! reduce dataset
df = df[(df.Sex == 'M') & 
        (df.Height > 1) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year >= 1960) &
        (df.Sport == 'Athletics')
        ]


def add_theese_and_csv(df, name, medal= True, div_avg= True):
    # * add diviation from average columns for given variables
    if div_avg:
        add.div_from_avg_per_year(df, vals)
    
    # * add MedalEarned column to dataframe
    if medal:
        add.medal_earned(df)
    
    # * saves df as csv
    df.to_csv(name + '.csv')


add_theese_and_csv(df, 'csv_with_columns-HWAM')
