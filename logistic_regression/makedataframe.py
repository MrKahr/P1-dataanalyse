import pandas as pd
import addvariables as add

# ! Get dataset and variables
filepath = 'Datasets/athlete_events.csv'
df = pd.read_csv(filepath)

vals = ['Height', 'Weight', 'Age']

# ! reduce dataset
def Reduction(df):
    df = df[(df.Event == 'Athletics Men\'s 100 metres') |
            (df.Event == 'Athletics Men\'s Long Jump') |
            (df.Event == 'Athletics Men\'s Shot Put') |
            (df.Event == 'Athletics Men\'s High Jump') |
            (df.Event == 'Athletics Men\'s 400 metres') |
            (df.Event == 'Athletics Men\'s 110 metres Hurdles') |
            (df.Event == 'Athletics Men\'s Discus Throw') |
            (df.Event == 'Athletics Men\'s Pole Vault') |
            (df.Event == 'Athletics Men\'s Javelin Throw') |
            (df.Event == 'Athletics Men\'s 1,500 metres')]
    
    df = df[(df.Height > 1) &
            (df.Age > 1) &
            (df.Weight > 1) &
            (df.Year >= 1960)
            ]
    
    return df


def Reduction2(df):
    df = df[(df.Height > 1) &
            (df.Age > 1) &
            (df.Weight > 1) &
            (df.Year >= 1960) &
            (df.Event == 'Athletics Men\'s Decathlon')
            ]
    
    return df


def AddTheese(df, name, medal= True, prev_med= True, div_avg= True, div_avg_event= False, reduce= True):
    # * add MedalEarned column to dataframe
    if medal:
        df = add.MedalEarned(df)
        print('Added medal_earned')
    
    # * add previous medal column to dataframe
    if prev_med:
        df = add.PreviousMedals(df)
        print('Added previous_medals')
    
    # * remove unneeded variables and create dummy
    df_noc = df['NOC']
    df_dummy = pd.get_dummies(df_noc)
    df_ = pd.concat([df, df_dummy], axis= 1)
    
    # * reduce dataframe
    if reduce:
        df_ = Reduction(df_)
        print('reduced')
    
    # * add diviation from average columns for given variables per year
    if div_avg:
        df_ = add.DeviationAverage(df_, vals)
        print('Added div_avg')
    
    # * add diviation from average columns for given variables per year per event
    if div_avg_event:
        df_ = add.DeviationAverageEvent(df_, vals)
        print('Added div_avg_event')
    
    df_d = df_.drop(['Name', 'Team', 'Games', 'Season', 'City', 'Sport', 'Medal', 'NOC', 'Event', 'Year', 'Sex'], axis= 1)
    
    # * saves df as csv
    df_d.to_csv(name + '.csv')


AddTheese(df, 'dec_sep_NOCdummy')
