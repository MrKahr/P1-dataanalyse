import pandas as pd
import numpy as np
import random

filepath = 'athlete_events.csv'

df = pd.read_csv(filepath)

events = ['Athletics Men\'s Shot Put', 
          'Athletics Men\'s Hammer Throw', 
          'Athletics Men\'s Discus Throw']

# Reduce dataframe
df_a = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Event == 'Athletics Men\'s Shot Put')]

df_b = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Event == 'Athletics Men\'s Hammer Throw')]

df_c = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Event == 'Athletics Men\'s Discus Throw')]

dfss = [df_a, df_b, df_c]
df = pd.concat(dfss)

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

print(df_0)


# add random number between 0 and 1 to all id
# Make sure there are same amount of 1 and 0
for i, row in df_0.iterrows():
    newVal = random.random()

    df_0.at[i,'Bonk'] = newVal

#print(len(df_0))
df_0 = df_0.drop(df_0[df_0.Bonk > 0.1].index)
#print(len(df_0))

dfs = [df_0, df_1]
df = pd.concat(dfs)

# Make test and train dataframes
for i, row in df.iterrows():
    newVal = random.random()

    df.at[i,'Bonk'] = newVal

df_test = df[(df.Bonk < 0.34)]
df_train = df[(df.Bonk >= 0.34)]

# Reduce dataframes
col_list = ['ID', 'Height', 'MedalValue', 'Weight', 'Age']

df_test = df_test[col_list]
df_train = df_train[col_list]

# Split X and Y dataframes
X_list = ['ID', 'Height', 'Age', 'Weight']
Y_list = ['ID', 'MedalValue']

X_train = df_train[X_list]
Y_train = df_train[Y_list]
X_test = df_test[X_list]
Y_test = df_test[Y_list]

# Create csv files
X_train.to_csv('X_train_HWAY.csv', index=False)
Y_train.to_csv('Y_train_HWAY.csv', index=False)
X_test.to_csv('X_test_HWAY.csv', index=False)
Y_test.to_csv('Y_test_HWAY.csv', index=False)

