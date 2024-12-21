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
# Define CSV path
path = r'C:\Users\guill\Downloads\data\ALL'

# List to save dataframe
list = []

# Iterate over each file
for archivo in os.listdir(path):
    if archivo.startswith("WID_data_") and archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, archivo), delimiter=';')
        list.append(df)

# Combine all dataframe
dfx = pd.concat(list, ignore_index=True)

# Data Manipulation
# ===================================================
# Filter dataframe
variable = ['sdiincj992', 'adiincj992']
variable2 = ['gdiincj992']
country = ['US', 'CL', 'NL', 'ZA', 'CN', 'IN', 'AU']
year = [2022]
df = dfx[dfx['variable'].isin(variable) & dfx['year'].isin(year)]

# DF Gini
dfg = dfx[dfx['variable'].isin(variable2) & dfx['year'].isin(year) & dfx['country'].isin(country)]
dfg = dfg[['country', 'value']]
dfg = dfg[['country', 'value']].rename(columns={'value': 'gini'})

# Clean dataframe
df = df[~df['percentile'].str.contains(r'\.', na=False)]
df['dif'] = df['percentile'].str.extract(r'p(\d+)p(\d+)').astype(int).apply(lambda x: x[1] - x[0], axis=1)
df = df[df['dif'] == 1]
df['percentile'] = df['percentile'].str.extract(r'p\d+p(\d+)').astype(int)
df = df.sort_values(by='percentile')

# Pivot dataframe
df['variable'] = df['variable'].replace({'sdiincj992': 'percentage', 'adiincj992': 'value', 'gdiincj992': 'gini'})
df = df[['variable', 'country', 'percentile', 'value']]
df = df.pivot_table(index=['country', 'percentile'], columns='variable', values='value')
df = df.reset_index()

# Calculate cummulative
df['percentile'] =  df['percentile'] / 100
df['value'] =  df.groupby(['country'])['value'].cumsum() / df.groupby(['country'])['value'].transform('sum')

# Merge regions 
df = df.merge(df_countries, how='left', left_on='country', right_on='ISO2')
df = df[['Region', 'Country_Abr','country', 'percentile', 'value']]
df = df[df['Region'].notna()]

# Create second dataframe
dfx = df.merge(dfg, on='country', how='inner')

# Dictionaire
region_colors = {
    'China': '#F4EE00',
    'India': '#FFC107',
    'Netherlands': '#00F26D',
    'Australia': '#79E9FF',
    'USA': '#FF0000',
    'Chile': '#FF8989',
    'South Africa': '#F2CEEF'
}

dfx["color"] = dfx["Country_Abr"].map(region_colors)
country_colors = dfx.groupby("country")["color"].first().to_dict()

# Data Visualization
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})
plt.figure(figsize=(10, 10))

# Basic Grey Plot Lines
sns.lineplot(
    data=df, 
    x="percentile", 
    y="value", 
    hue="country",
    linewidth=0.4,
    alpha=0.5,
    palette=['#808080']
).legend_.remove()

# Black Shadow Plot Lines
sns.lineplot(
    data=dfx, 
    x="percentile", 
    y="value", 
    hue="country",
    linewidth=2.25,
    alpha=1,
    palette=['black']
).legend_.remove()

# Color Plot Lines
sns.lineplot(
    data=dfx, 
    x="percentile", 
    y="value", 
    hue="country",
    linewidth=1.5,
    alpha=1,
    palette=country_colors
).legend_.remove()

# Add Inequality lines
plt.plot([0, 1], [0, 1], color="gray", linestyle="-", linewidth=1)

# Configuración del gráfico
plt.text(0, 1.05, 'Global Income Distribution', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'Lorenz Curve Comparision Across Countries', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income (%)', fontsize=10, fontweight='bold')
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

# Add Notes
plt.text(0, -0.12, 'Income: Post-tax national income is the sum of primary incomes over all sectors (private and public), minus taxes.', 
    transform=plt.gca().transAxes, 
    fontsize=7,
    fontstyle='italic',
    color='gray')

 # Add Year label
formatted_date = 2022
plt.text(1, 1.06, f'{formatted_date}',
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Legend Dataframe
gini_values = dfx.groupby(['country', 'Country_Abr'])['gini'].mean().reset_index()
gini_values_sorted = gini_values.sort_values(by='gini', ascending=True)

# Create a custom legend
custom_legend = [
    plt.Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor=country_colors[country],
        lw=0,
        label=f"{gini_values_sorted.loc[gini_values_sorted['country'] == country, 'Country_Abr'].values[0]}: {gini_values_sorted.loc[gini_values_sorted['country'] == country, 'gini'].values[0]:.2f}"
    )
    for country in gini_values_sorted['country'].unique() if country in country_colors
]

# Add the legend
legend = plt.legend(handles=custom_legend, title='Region', title_fontsize='9', fontsize='8', loc='upper left')
plt.setp(legend.get_title(), fontweight='bold')

# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_WID_Global_Lorenz_Curves.png', format='png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()