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
path = r'C:\Users\guillem.maya\Downloads\data\X'

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
variable = ['shwealj992', 'ahwealj992']
variable2 = ['ghwealj992']
country = ['US', 'CL', 'DE', 'ZA', 'CN', 'IN', 'AU']
year = [2022]
df = dfx[dfx['variable'].isin(variable) & dfx['year'].isin(year)]

# Clean dataframe
df = df[~df['percentile'].str.contains(r'\.', na=False)]
df['dif'] = df['percentile'].str.extract(r'p(\d+)p(\d+)').astype(int).apply(lambda x: x[1] - x[0], axis=1)
df = df[df['dif'] == 1]
df['percentile'] = df['percentile'].str.extract(r'p\d+p(\d+)').astype(int)
df = df.sort_values(by='percentile')

# Pivot dataframe
df['variable'] = df['variable'].replace({'shwealj992': 'percentage', 'ahwealj992': 'value', 'ghwealj992': 'gini'})
df = df[['variable', 'country', 'percentile', 'value']]
df = df.pivot_table(index=['country', 'percentile'], columns='variable', values='value')
df = df.reset_index()

# Calculate cummulative
df['percentile'] =  df['percentile'] / 100
df['value'] =  df.groupby(['country'])['value'].cumsum() / df.groupby(['country'])['value'].transform('sum')

# Merge regions 
df = df.merge(df_countries, how='left', left_on='country', right_on='ISO2')
df = df[['Region', 'country', 'percentile', 'value']]
df = df[df['Region'].notna()]

dfx = df[df['country'].isin(country)]

# Dictionaire
region_colors = {
    'CN': '#cc9d0e',
    'IN': '#FFC107',
    'DE': '#00B050',
    'AU': '#0F9ED5',
    'US': '#a60707',
    'CL': '#FF0000',
    'ZA': '#FF6F00'
}

dfx["color"] = dfx["country"].map(region_colors)
country_colors = dfx.groupby("country")["color"].first().to_dict()

# Data Visualization
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Crear el gráfico
plt.figure(figsize=(10, 10))
sns.lineplot(
    data=df, 
    x="percentile", 
    y="value", 
    hue="country",
    linewidth=0.4,
    alpha=0.5,
    palette=['#808080']
).legend_.remove()

sns.lineplot(
    data=dfx, 
    x="percentile", 
    y="value", 
    hue="country",
    linewidth=2,
    alpha=1,
    palette=country_colors
).legend_.remove()

# Añadir la línea de igualdad
plt.plot([0, 1], [0, 1], color="gray", linestyle="-", linewidth=1)

# Configuración del gráfico
plt.text(0, 1.05, 'Wealth Distribution', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'By countries in 2022', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Adjust grid and layout
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Add Data Source
plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

 # Add Year label
formatted_date = 2022
plt.text(1, 1.06, f'{formatted_date}',
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Mostrar el gráfico
plt.show()