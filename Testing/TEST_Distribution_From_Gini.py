# Libraries
# =======================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# Input Data
# =======================================================
# Data Example
countries = [
    {"country": "ESP", "income": 500, "gini": 0.8, "population": 1000000}
]

# Functions
# =======================================================
# Function to calculate sigma from Gini
def gini_to_sigma(gini):
    return np.sqrt(2) * norm.ppf((gini + 1) / 2)

# Function to simulate distribution
def simulate_income_distribution(income, gini, population):
    sigma = gini_to_sigma(gini)
    mu = np.log(income) - (sigma**2) / 2
    incomes = lognorm(s=sigma, scale=np.exp(mu)).rvs(size=population)
    return incomes

# Function to calculate Gini
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

# Data Manipulation
# =======================================================
# Create Dataframe 
data = []

for country in countries:
    incomes = simulate_income_distribution(country["income"], country["gini"], country["population"])
    df_country = pd.DataFrame({
        "Country": country["country"],
        "Income": incomes
    })
    data.append(df_country)

# Concatenate Dataframes
df = pd.concat(data, ignore_index=True)

# Sorting and adding columns
df = df.sort_values(by="Income", ascending=True)
df['Log_Income'] = np.log(df['Income'])
df['Average_Income'] = df['Income'].mean()
df['Median_Income'] = df['Income'].median()
df['Gini'] = df['Income'].apply(lambda x: gini(df['Income']))

# Calculate quantiles
quantile_min = df['Income'].quantile(0.01)
quantile_max = df['Income'].quantile(0.99)

# Show Data
print(df)

# Data Visualization
# =======================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Plot income distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Income'], color='lightblue', bins=500, kde=False)


# Title and labels
plt.text(0, 1.05, 'Income Distribution', fontsize=12, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'Simulation from input data', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Income per capita', fontsize=10, fontweight='bold')
plt.ylabel('Density', fontsize=10, fontweight='bold')
plt.xlim(0, right=quantile_max)

# Show the plot!
plt.show()

