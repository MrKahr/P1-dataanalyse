import pandas as pd
import Add_columns_to_df as add

# ! Get dataset
filepath = 'test.csv'
df = pd.read_csv(filepath)

#df_list = ['ID','Year','MedalEarned']

#df = df[df_list]

#print(df_narrow)

print(add.previous_medals(df))