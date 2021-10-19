import numpy as np


# ! Converts Medal index from string to integer
def medal_con(df):
    conditions = [(df['Medal'] == 'Gold'),
                (df['Medal'] == 'Silver'),
                (df['Medal'] == 'Bronze'),
                (df['Medal'] == 'NA')]

    values = [3,2,1,0]

    df['MedalValue'] = np.select(conditions,values)

    return df