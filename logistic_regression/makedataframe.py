import pandas as pd
import addvariables as add

# ! Get dataset and variables
filepath = 'Datasets/athlete_events.csv'
df = pd.read_csv(filepath)

vals = ['Height', 'Weight', 'Age']

# ! reduce dataset
def Reduction(df):
    df = df[(df.Height > 1) &
            (df.Age > 1) &
            (df.Weight > 1) &
            (df.Year >= 1960)
            ]
    '''df = df[(df.Event == 'Athletics Men\'s 100 metres') |
            (df.Event == 'Athletics Men\'s Long Jump') |
            (df.Event == 'Athletics Men\'s Shot Put') |
            (df.Event == 'Athletics Men\'s High Jump') |
            (df.Event == 'Athletics Men\'s 400 metres') |
            (df.Event == 'Athletics Men\'s 110 metres Hurdles') |
            (df.Event == 'Athletics Men\'s Discus Throw') |
            (df.Event == 'Athletics Men\'s Pole Vault') |
            (df.Event == 'Athletics Men\'s Javelin Throw') |
            (df.Event == 'Athletics Men\'s 1,500 metres')]'''
    return df


def Reduction2(df):
    df = df[(df.Height > 1) &
            (df.Age > 1) &
            (df.Weight > 1) &
            (df.Year >= 1960)
            ]
    '''df = df[(df.Event == 'Athletics Men\'s Decathlon')]'''
    
    return df


def AddTheese(df, name, medal= True, prev_med= True, div_avg= True, div_avg_event= True, reduce= True, strength= True):
    # * Add MedalEarned column to dataframe
    if medal:
        df = add.MedalEarned(df)
        print('Added medal_earned')
    
    # * Add previous medal column to dataframe
    if prev_med:
        df = add.PreviousMedals(df)
        print('Added previous_medals')
    
    # * Reduce events
    df = df[(df.Event == 'Athletics Men\'s Decathlon')]
    
    # * Add NOC strength
    if strength:
        df = add.NOCStrength(df)
        print('NOC strength added')
    
    # * Reduce dataframe
    if reduce:
        df = Reduction(df)
        print('reduced')
    
    # * Add diviation from average columns for given variables per year
    if div_avg:
        df = add.DeviationAverage(df, vals)
        print('Added div_avg')
    
    # * Add diviation from average columns for given variables per year per event
    if div_avg_event:
        df = add.DeviationAverageEvent(df, vals)
        print('Added div_avg_event')
    
    # * Saves df as csv
    df.to_csv(name + '.csv')


AddTheese(df, 'decathlon_data', div_avg_event= False)
