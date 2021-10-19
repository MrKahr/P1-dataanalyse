import pandas as pd
import numpy as np

import function_folder.find_sport_and_event as find
import function_folder.MedalValue_converter as mdl

file_path = 'athlete_events.csv'
df = pd.read_csv(file_path)


def find_ex(df): 
    df_women = df[(df.Sex == 'F')]
    df_sport = df_women[(df.Sport == 'Athletics')]
    
    pd.set_option('display.max_rows',100)
    print(find.find_event(df_sport))


def mdl_ex(df):
    df_wmn_ath = df[(df.Sex == 'F') & (df.Sport == 'Athletics')]
    
    df_mdl_con = mdl.medal_con(df_wmn_ath)
    
    mdl_sum = 0
    
    for i, row in df_mdl_con.iterrows():
        mdl_sum += row['MedalValue']
    
    print(f'total sum of medal points earned by women in athletics: {mdl_sum}')

mdl_ex(df)
