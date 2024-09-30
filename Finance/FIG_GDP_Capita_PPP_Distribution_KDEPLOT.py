# Libraries
# =====================================================================
import requests
import wbgapi as wb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Data Extraction (Countries)
# =====================================================================
# Extract JSON and bring data to a dataframe
url = 'https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json'
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df_countries = df.rename(columns={'index': 'ISO3'})

# Data Extraction - IMF (1980-2030)
# =====================================================================
#Parametro
parameters = ['NGDPD', 'PPPGDP', 'LP']

# Create an empty list
records = []

# Iterar sobre cada parámetro
for parameter in parameters:
    # Request URL
    url = f"https://www.imf.org/external/datamapper/api/v1/{parameter}"
    response = requests.get(url)
    data = response.json()
    values = data.get('values', {})

    # Iterate over each country and year
    for country, years in values.get(parameter, {}).items():
        for year, value in years.items():
            records.append({
                'Parameter': parameter,
                'ISO3': country,
                'Year': int(year),
                'Value': float(value)
            })
    
# Create dataframe
df_imf = pd.DataFrame(records)

# Pivot Parameter to columns and filter nulls
df_imf = df_imf.pivot(index=['ISO3', 'Year'], columns='Parameter', values='Value').reset_index()

# Filter after 2024
df_imf = df_imf[df_imf['Year'] == 2024]

# Data Manipulation
# =====================================================================
# Concat and filter dataframes
df = df_imf.dropna(subset=['NGDPD', 'PPPGDP', 'LP'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['Region', 'ISO3', 'Country', 'Cod_Currency', 'Year', 'NGDPD', 'PPPGDP', 'LP']]
df = df[df['Cod_Currency'].notna()]

# Calculate PPP
df = df.groupby(['Region', 'ISO3', 'Year'])[['NGDPD', 'PPPGDP', 'LP']].sum()
df = df.reset_index()
df['PPP'] = df['NGDPD'] / df['PPPGDP']
df['NGDPDPC'] = df['NGDPD'] / df['LP']
df['PPPPC'] = df['PPPGDP'] / df['LP']

# Calculate Average Weight and Percent
df['AVG_Weight'] = df.groupby('Year')['NGDPDPC'].transform(lambda x: np.average(x, weights=df.loc[x.index, 'LP']))
df['Percent'] = df['NGDPD'] / df.groupby('Year')['NGDPD'].transform('sum')

# Filtering
df = df[df['NGDPDPC'] < df['AVG_Weight'] * 60 ]
df = df[df['PPP'] < 1.2]

print(df)

# Data Visualization
# =====================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

#Palette of colors
custom_area = {
    'Asia': '#fff3d0',
    'Europe': '#ccdccd',
    'Oceania': '#90a8b7',
    'Americas': '#fdcccc',
    'Africa': '#ffe3ce'
}
custom_area_m = df['Region'].map(custom_area)

# Custom palette line
custom_line = {
    'Asia': '#FFC107',
    'Europe': '#004d00',
    'Oceania': '#003366',
    'Americas': '#FF0000',
    'Africa': '#FF6F00'
}
custom_line_m = df['Region'].map(custom_line)

# Filtering
usa = df.loc[df['ISO3'] == 'USA', 'NGDPDPC'].max() * 1.1
median = df.loc[df['ISO3'] == 'USA', 'AVG_Weight'].max()

# Create scatter plot
plt.figure(figsize=(10,10))
plt.scatter(df['PPP'], df['NGDPDPC'], s=df['NGDPD']/8, edgecolor=custom_line_m, facecolor=custom_area_m, linewidth=0.5)

# Add title labels
plt.text(0, 1.05, 'Relationship of GDP Per Capita and Price Levels', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'Comparing Inequalities Between PPP and Market Exchange Rates by Country', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('GAP Between PPP and Exchange Rate', fontsize=10, fontweight='bold')
plt.ylabel('GDP Per Capita ($US)', fontsize=10, fontweight='bold')
plt.xlim(0, 1.2)
plt.ylim(0, usa)
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}k'))
plt.gca().set_yticks(np.linspace(0, usa, 7))

# Add line trend
z = np.polyfit(df['PPP'], df['NGDPDPC'], 2, w=df['NGDPD'])
p = np.poly1d(z)
x_range = np.linspace(df['PPP'].min(), df['PPP'].max(), 100)  # 100 puntos entre el mínimo y máximo
y_range = p(x_range)
plt.plot(x_range, y_range, color='darkred', linewidth=0.5)

# Add line median
plt.axhline(y=median, color='darkred', linewidth=0.25, linestyle='--')

# Add text median
plt.text(0.2, median*1.15, f'Median: {median:,.2f}k',
    fontsize=9, ha='right', va='top',
    fontweight='bold', color='darkred')

# Element legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region, alpha=0.35)
    for region, color in custom_line.items()
]

# Create legend
legend = plt.legend(handles=legend_elements, title='Region', title_fontsize='10', fontsize='9', loc='upper left')
plt.setp(legend.get_title(), fontweight='bold')

# Fill red and green area
plt.fill_betweenx(y=[0, usa], x1=0, x2=1, color='red', alpha=0.04)
plt.fill_betweenx(y=[0, usa], x1=1, x2=1.25, color='green', alpha=0.04)

# Add Year label
plt.text(0.95, 1.06, df['Year'].max(),
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Add Data Source
plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Save the figure
plt.savefig('C:/Users/guillem.maya/Desktop/FIG_PPP_Inequalities.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
