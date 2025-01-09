# Libraries
# ===================================================
import requests
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.patheffects as patheffects

# Data Extraction (Countries)
# =====================================================================
# Extract JSON and bring data to a dataframe
url = 'https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json'
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df_countries = df.rename(columns={'index': 'ISO3'})

# Data Extraction
# ===================================================
# URL GitHub
url = "https://raw.githubusercontent.com/guillemmaya92/Python/main/Data/WID_Percentiles.parquet"

# Extract data from parquet
df = pd.read_parquet(url, engine='pyarrow')

# Filter year
df = df[df['year'] == 2021]

# Data Manipulation
# ===================================================
# Select between income and wealth
selection = 'Wealth'

# Calculate cummulative
df['percentile'] =  df['percentile'] / 100
df['income'] =  df['income'] / 100
df['wealth'] =  df['wealth'] / 100
df['income_cum'] =  df.groupby(['country'])['income'].cumsum() / df.groupby(['country'])['income'].transform('sum')
df['wealth_cum'] =  df.groupby(['country'])['wealth'].cumsum() / df.groupby(['country'])['wealth'].transform('sum')
df['value_cum'] = df['income_cum'] if selection == 'income' else df['wealth_cum']

# Countries
dfc = df.merge(df_countries, how='left', left_on='country', right_on='ISO2')
dfc = dfc[dfc['Region'].notna()]
dfc = dfc[['country', 'percentile', 'value_cum']]

# World
dfw = df[df['country'] == "WO"]
dfw = dfw[['country', 'percentile', 'value_cum']]
dfw['country'] = 'Inter-Countries'

print(df)

# Data Visualization
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})
plt.figure(figsize=(10, 10))

# Basic Grey Plot Lines
sns.lineplot(
    data=dfc, 
    x="percentile", 
    y="value_cum", 
    hue="country",
    linewidth=0.4,
    alpha=0.5,
    palette=['#808080']
).legend_.remove()

# Black Shadow Plot Lines
sns.lineplot(
    data=dfw, 
    x="percentile", 
    y="value_cum", 
    hue="country",
    linewidth=2.25,
    alpha=1,
    palette=['black']
).legend_.remove()

# Color Plot Lines
sns.lineplot(
    data=dfw, 
    x="percentile", 
    y="value_cum", 
    hue="country",
    linewidth=1.5,
    alpha=1,
    palette=['#FF0000']
).legend_.remove()

# Add Inequality lines
plt.plot([0, 1], [0, 1], color="gray", linestyle="-", linewidth=1)

# Configuración del gráfico
plt.text(0, 1.05, f'Global {selection} Distribution', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'Lorenz Curve Comparision Across Countries', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel(f'Cumulative {selection} (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Adjust grid and layout
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Add Data Source
plt.text(0, -0.1, 'Data Source: World Inequality Database (WID)', 
    transform=plt.gca().transAxes, 
    fontsize=8,
    fontweight='bold',
    color='gray')

# Variable notes
noteincome = 'Income: Post-tax national income is the sum of primary incomes over all sectors (private and public), minus taxes.'
notewealth = 'Wealth: Total value of non-financial and financial assets held by households, minus their debts.'
note = noteincome if selection == 'income' else notewealth

# Add Notes
plt.text(0, -0.12, note, 
    transform=plt.gca().transAxes, 
    fontsize=7,
    fontstyle='italic',
    color='gray')

# Add Author
plt.text(0.85, -0.1, '@guillemmaya.com', 
    transform=plt.gca().transAxes, 
    fontsize=9,
    fontstyle='italic',
    color='#212121')

 # Add Year label
formatted_date = 2022
plt.text(1, 1.06, f'{formatted_date}',
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Gini Values
ginic = 0.64
giniw = 0.75

# Create custom lines
intra_line = mlines.Line2D([], [], color='#808080', label=f'Gini Intra-Countries: {ginic}', linewidth=2)
inter_line = mlines.Line2D([], [], color='#FF0000', label=f'Gini Inter-Countries: {giniw}', linewidth=2)
inter_line.set_path_effects([patheffects.withStroke(linewidth=4, foreground='black')])
inter_circle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#FF0000', markeredgecolor='black', markersize=8, label='Inter-Countries', linewidth=0)

# Add custom legend
plt.legend(handles=[intra_line, inter_line])

# Save the figure
plt.savefig('C:/Users/guillem.maya/Desktop/FIG_WID_Global_Lorenz_Curves.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot!
plt.show()

