# Thor
import seaborn as sns
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# Make the dataframe
filePath = r"C:\Users\Thor\Desktop\Uni\archive\athlete_events.csv"
df = pd.read_csv(filePath)


# The variables you want to plot
xVariable = 'Year'
yVariable = 'Height'

# Sort the dataset you want to plot, in this case only male basketball players
df = df[df.Sport == "Basketball"]
df = df[df.Sex == "M"]

# drop NA values because i doesn't work when you need to get linreg.
df.dropna(subset = [xVariable], inplace=True)
df.dropna(subset = [yVariable], inplace=True)


# calculate r_values, slope etc.
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(df[xVariable], df[yVariable])

# round the variables so when they are put on the graph it's not too long
shortR2 = round(r_value, 2)

shortSlope = round(slope, 2)
shortIntercept = round(intercept, 2)


# Make the regplot
ax = sns.regplot(data = df, x = xVariable, y= yVariable)

# Insert the text to show R2 and the equation
ax.text(.05, 0.95, f'r2 = {shortR2}, y = {shortSlope}x + {shortIntercept}',
            transform=ax.transAxes)

# Show the graph on screen
plt.show()