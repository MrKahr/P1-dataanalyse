import addvariables as add
import pandas as pd

# ! Get dataset and variables
filepath = 'Datasets/dec_sep_MPHWA.csv'
df = pd.read_csv(filepath)

add.NOCStrength(df)
