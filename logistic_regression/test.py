import pandas as pd
import Add_columns_to_df as add

# ! Get dataset
filepath = 'csv_with_columns-HWAM.csv'
df = pd.read_csv(filepath)

df_list = ['ID','Year', 'MedalEarned']

df_narrow = df[df_list]

#print(df_narrow)

print(add.medal_earned(df_narrow))