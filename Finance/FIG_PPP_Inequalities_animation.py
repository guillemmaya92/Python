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
parameters = ['PPPPC', 'NGDPDPC', 'LP']

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
df_imf = df_imf[df_imf['Year'] >= 1999]

# Data Manipulation
# =====================================================================
# Concat and filter dataframes
df = df_imf.dropna(subset=['NGDPDPC', 'PPPPC', 'LP'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['Region', 'ISO3', 'Country', 'Cod_Currency', 'Year', 'NGDPDPC', 'PPPPC', 'LP']]
df = df[df['Cod_Currency'].notna()]

# Calculate PPP
df = df.groupby(['Region', 'ISO3', 'Year'])[['NGDPDPC', 'PPPPC', 'LP']].sum()
df = df.reset_index()
df['PPP'] = df['NGDPDPC'] / df['PPPPC']

print(df)

# Data Visualization
# =====================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Palette of colors
custom_area = {
    'Asia': '#fff3d0',
    'Europe': '#ccdccd',
    'Oceania': '#90a8b7',
    'Americas': '#fdcccc',
    'Africa': '#ffe3ce'
}
df['custom_area'] = df['Region'].map(custom_area)

# Custom palette line
custom_line = {
    'Asia': '#FFC107',
    'Europe': '#004d00',
    'Oceania': '#003366',
    'Americas': '#FF0000',
    'Africa': '#FF6F00'
}
df['custom_line'] = df['Region'].map(custom_line)

fig, ax = plt.subplots(figsize=(10, 10))

def update(year):
    # Filtrar datos por año
    ax.clear()
    filtered_df = df[df['Year'] == year]
    usa = filtered_df.loc[filtered_df['ISO3'] == 'USA', 'NGDPDPC'].max() * 1.1

    # Crear gráfico de dispersión
    scatter = ax.scatter(filtered_df['PPP'], filtered_df['NGDPDPC'], s=filtered_df['LP']/2,
                         edgecolor=filtered_df['custom_line'], facecolor=filtered_df['custom_area'],
                         linewidth=0.5)

    # Añadir títulos y etiquetas
    ax.text(0, 1.05, 'Income Inequality', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
    ax.text(0, 1.02, 'Differences between PPP and market exchanges rates', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
    ax.set_xlabel('GAP Between PPP and Exchange Rate', fontsize=10, fontweight='bold')
    ax.set_ylabel('GDP Per Capita ($US)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, usa)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, linestyle='-', color='grey', linewidth=0.08)
    ax.set_yticks(np.linspace(0, usa, 7))

    # Añadir línea de tendencia
    z = np.polyfit(filtered_df['PPP'], filtered_df['NGDPDPC'], 1)
    p = np.poly1d(z)
    ax.plot(filtered_df['PPP'], p(filtered_df['PPP']), color='darkred', linewidth=0.5)

    # Rellenar áreas roja y verde
    ax.fill_betweenx(y=[0, usa], x1=0, x2=1, color='red', alpha=0.04)
    ax.fill_betweenx(y=[0, usa], x1=1, x2=1.25, color='green', alpha=0.04)

# Configurar animación
years = sorted(df['Year'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=250, blit=False)

# Guardar la animación :)
ani.save('C:/Users/guillem.maya/Desktop/FIG_PPP_Inequalities.webp', writer='imagemagick', fps=2)

# Mostrar el gráfico
plt.show()