import pandas as pd
import function_folder.find_sport_and_event as find

file_path = 'athlete_events.csv'
df = pd.read_csv(file_path)

df_women = df[(df.Sex == 'F')]
df_sport = df_women[(df.Sport == 'Athletics')]

pd.set_option('display.max_rows',100)
print(find.find_sport(df_women))