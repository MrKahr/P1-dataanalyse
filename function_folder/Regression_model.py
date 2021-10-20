# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:28:21 2021

@author: Daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.metrics import r2_score

filepath = r'C:\Users\danie\Desktop\Alt\Skole\AAU\1. semester\P1 projekt\Data\athlete_events.csv'

df = pd.read_csv(filepath)


df = df[(df.Year > 1945) & (df.Sport == 'Basketball') & (df.Sex == 'M')]

conditions = [(df['Medal'] == 'Gold'),
              (df['Medal'] == 'Silver'),
              (df['Medal'] == 'Bronze'),
              (df['Medal'] == 'NA')]

values = [3,2,1,0]

df['MedalValue'] = np.select(conditions,values)

df2 = df.groupby('Height')['MedalValue'].mean().reset_index()

x = df2['Height']
y = df2['MedalValue']

mymodel = np.poly1d(np.polyfit(x,y,2))

myline = np.linspace(160,230,100)

plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()


print(r2_score(y,mymodel(x)))
print(mymodel)