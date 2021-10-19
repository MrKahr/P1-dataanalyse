import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'athlete_events.csv'
df = pd.read_csv(file_path)

# ! Adds a MedalValue column
def medal_con(df):
    conditions = [(df['Medal'] == 'Gold'),
                (df['Medal'] == 'Silver'),
                (df['Medal'] == 'Bronze'),
                (df['Medal'] == 'NA')]

    values = [3,2,1,0]

    df['MedalValue'] = np.select(conditions,values)

    return df


# ! Plot country vs. country in given event
# Cut dataset down to given event
df_sport = df[(df.Sport == 'Athletics') & (df.Year > 1945)]

# Cut dataset down to only have given countrys
df_NOC = df[(df.NOC == 'NOR') | (df.NOC == 'SWE') |(df.NOC == 'DEN')]

medal_con(df_NOC)

group = df_NOC.groupby(['Year', 'NOC'])['MedalValue'].mean()

group = group.reset_index()

#print(group)

sns.lineplot(data = group, x = 'Year', 
             y = 'MedalValue', hue = 'NOC', 
             style = 'NOC', markers=['o', 'o', 'o'])

'''
sns.barplot(data = group, x = 'Year', 
             y = 'MedalValue', hue = 'NOC')
'''

plt.show()