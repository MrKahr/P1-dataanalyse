import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from seaborn.categorical import boxenplot
import random


filepath = 'athlete_events.csv'

df = pd.read_csv(filepath)

# Reduce dataframe
df = df[(df.Sport == 'Athletics') & 
        (df.Sex == 'M') & 
        (df.Year > 1940) &
        (df.Height > 130)]

# Make MedalValue column
conditions = [(df['Medal'] == 'Gold'),
              (df['Medal'] == 'Silver'),
              (df['Medal'] == 'Bronze'),
              (df['Medal'] == 'NA')]

values = [1,1,1,0]

df['MedalValue'] = np.select(conditions,values)

# Split dataframe into won a medal and didnt win a medal
df_0 = df[(df.MedalValue == 0)]
df_1 = df[(df.MedalValue == 1)]

print(len(df_1))


# add random number between 0 and 1 to all id
# Make sure there are same amount of 1 and 0
for i, row in df_0.iterrows():
    newVal = random.random()

    df_0.at[i,'Bonk'] = newVal

df_0 = df_0[(df_0.Bonk <= (len(df_1) / 100))]

dfs = [df_0, df_1]
df = pd.concat(dfs)

# Make test and train dataframes
for i, row in df.iterrows():
    newVal = random.random()

    df.at[i,'Bonk'] = newVal

df_test = df[(df.Bonk < 0.34)]
df_train = df[(df.Bonk >= 0.34)]

# Reduce dataframes
col_list = ['ID', 'Height', 'MedalValue']

df_test = df_test[col_list]
df_train = df_train[col_list]

# Split X and Y dataframes
X_list = ['ID', 'Height']
Y_list = ['ID', 'MedalValue']

X_train = df_train[X_list]
Y_train = df_train[Y_list]
X_test = df_test[X_list]
Y_test = df_test[Y_list]

# Create csv files
#X_train.to_csv('X_train.csv', index=False)
#Y_train.to_csv('Y_train.csv', index=False)
#X_test.to_csv('X_test.csv', index=False)
#Y_test.to_csv('Y_test.csv', index=False)

