'''
This file is for logging the average accuracy of the model for a given dataset
'''
import pandas as pd

filepath = 'athlete_events.csv'
df = pd.read_csv(filepath)

filepath_2 = 'csv_with_columns-HWMA.csv'
df_HWAM = pd.read_csv(filepath)

filepath_3 = 'df_MPHWA_Athletics.csv'
df_MPHWA = pd.read_csv(filepath_3)

# ? all results underneath are run with theese parameters:
# ? run_more(times = 50, iterations= 5000, learning_rate= 0.03)

# ! Mens atheletics af 1960 
# ! with diviation of Height/Weight/Age from average per year
# ! and MedalEarned
# ! and PreviousMedals
# length of df_1: 1395
# the average accuracy of the model over 50 iterations is:  62.4 %
# the lowest accuracy of the model over 50 iterations is 57.45 %
# the highest accuracy of the model over 50 iterations is 67.34 %
dfdf_MPHWA = df_MPHWA[(df_MPHWA.Sex == 'M') & 
                (df_MPHWA.Height > 1) &
                (df_MPHWA.Age > 1) &
                (df_MPHWA.Weight > 1) & 
                (df_MPHWA.Year >= 1960) &
                (df_MPHWA.Sport == 'Athletics')
                ]

X_list = ['ID',
          'PreviousMedals', 
          'Height_div_avg', 
          'Weight_div_avg', 
          'Age_div_avg'
          ]

# ? all results underneath are run with theese parameters:
# ? run_more(times = 50, iterations= 3500, learning_rate= 0.0002)

# ! Mens atheletics af 1960 
# ! with diviation of Height/Weight/Age from average per year
# ! and MedalEarned
# length of df_1: 1395
# the average accuracy of the model over 50 iterations is:  58.26 %
# the lowest accuracy of the model over 50 iterations is 54.33 %
# the highest accuracy of the model over 50 iterations is 62.46 %
df_HWAM = df_HWAM[(df_HWAM.Sex == 'M') & 
        (df_HWAM.Height > 1) &
        (df_HWAM.Age > 1) &
        (df_HWAM.Weight > 1) & 
        (df_HWAM.Year >= 1960) &
        (df_HWAM.Sport == 'Athletics')
        ]

X_list = ['ID', 'Height_div_avg', 'Weight_div_avg', 'Age_div_avg']
Y_list = ['ID', 'MedalEarned']

# ! Mens atheletics after 1960
# length of df_1: 1395
# the average accuracy of the model over 50 iterations is:  56.32 %
# the lowest accuracy of the model over 50 iterations is 52.48 %
# the highest accuracy of the model over 50 iterations is 60.71 %
df = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year >= 1960) &
        (df.Sport == 'Athletics')
        ]

X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']

# ! Mens atheletics after 1945
# length of df_1: 1641
# the average accuracy of the model over 50 iterations is:  55.72 %
# the lowest accuracy of the model over 50 iterations is 52.11 %
# the highest accuracy of the model over 50 iterations is 60.08 %
df = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year >= 1945) &
        (df.Sport == 'Athletics')
        ]

X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']

# ! Mens atheletics
# length of df_1: 2409
# the average accuracy of the model over 50 iterations is:  54.49 %
# the lowest accuracy of the model over 50 iterations is 51.48 %
# the highest accuracy of the model over 50 iterations is 57.2 %
df = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Sport == 'Athletics')
        ]

X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']

# ! Mens Shot Put, Hammer Throw and Discus Throw
# length of df_1: 236
# the average accuracy of the model over 50 iterations is:  51.07 %
# the lowest accuracy of the model over 50 iterations is 43.55 %
# the highest accuracy of the model over 50 iterations is 62.5 %
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

# Concatinate dataframes
dfss = [df_a, df_b, df_c]
df = pd.concat(dfss)

X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']

# ! Mens Shot Put, Hammer Throw and Discus Throw after 1960
# length of df_1: 125
# the average accuracy of the model over 50 iterations is:  53.39 %
# the lowest accuracy of the model over 50 iterations is 37.84 %
# the highest accuracy of the model over 50 iterations is 63.83 %
df_a = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year > 1960) &
        (df.Event == 'Athletics Men\'s Shot Put')]

df_b = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year > 1960) &
        (df.Event == 'Athletics Men\'s Hammer Throw')]

df_c = df[(df.Sex == 'M') & 
        (df.Height > 130) &
        (df.Age > 1) &
        (df.Weight > 1) & 
        (df.Year > 1960) &
        (df.Event == 'Athletics Men\'s Discus Throw')]

# Concatinate dataframes
dfss = [df_a, df_b, df_c]
df = pd.concat(dfss)

X_list = ['ID', 'Height', 'Weight', 'Age']
Y_list = ['ID', 'MedalValue']