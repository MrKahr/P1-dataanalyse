import pandas as pd
import addcolumns as add

# ! Get dataset and variables
filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

vals = ['Height', 'Weight', 'Age']

# ! reduce dataset
def Reduction(df):
    df = df[(df.Sex == 'M') & 
            (df.Height > 1) &
            (df.Age > 1) &
            (df.Weight > 1) & 
            (df.Year >= 1960) &
            (df.Sport == 'Athletics')
            ]
    
    return df


def AddTheese(df, name, medal= True, prev_med= True, div_avg= True, reduce= True):
    # * add MedalEarned column to dataframe
    if medal:
        df = add.MedalEarned(df)
        print('Added medal_earned')
    
    # * add previous medal column to dataframe
    if prev_med:
        df = add.PreviousMedals(df)
        print('Added previous_medals')
    
    # * reduce dataframe
    if reduce:
        df = Reduction(df)
        print('reduced')
    
    # * add diviation from average columns for given variables per year
    if div_avg:
        df = add.DeviationAverage(df, vals)
        print('Added div_avg')
    
    # * saves df as csv
    df.to_csv(name + '.csv')


AddTheese(df, 'df_MPHWA_Athletics_3')
