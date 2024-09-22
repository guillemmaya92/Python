# Libraries
# =====================================================================
import requests
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

# Data Extraction (IMF)
# =====================================================================
#Parametro
parameters = ['LP', 'PPPPC']

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

# Data Manipulation
# =====================================================================
# Pivot Parameter to columns and filter nulls
df = df_imf.pivot(index=['ISO3', 'Year'], columns='Parameter', values='Value').reset_index()
df = df.dropna(subset=['LP', 'PPPPC'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['ISO3', 'Country', 'Year', 'LP', 'PPPPC', 'Analytical', 'Region']]
df = df[df['Region'].notna()]

# Filter nulls and order
df = df.sort_values(by=['Year', 'PPPPC'])

# Copy a df sample to calculate a median
columns = df.columns
df = np.repeat(df.values, df['LP'].astype(int) * 10, axis=0)
df = pd.DataFrame(df, columns=columns)

# Data Visualization
# =====================================================================
# Seaborn figure style
sns.set(style="whitegrid")

# Create a palette
fig, ax = plt.subplots(figsize=(16, 9))

def update(year):
    ax.clear()
    df_filtered = df[df['Year'] == year]
 
    # Calculation values   
    max_value = df_filtered[df_filtered['ISO3'] == 'USA']['PPPPC'].max()
    mean_value = df_filtered['PPPPC'].mean()

    # Custom palette
    custom = {
        'Asia': '#FFC107',
        'Europe': '#004d00',
        'Oceania': '#003366',
        'Americas': '#FF0000',
        'Africa': '#FF6F00'
    }
    
    # Region Order
    order_region = ['Asia', 'Africa', 'Americas', 'Europe', 'Oceania'] 

    # Create kdeplot area and lines
    sns.kdeplot(data=df_filtered, x="PPPPC", hue="Region", hue_order=order_region, multiple="stack", alpha=0.2, palette=custom, fill=True, linewidth=1, linestyle='-', ax=ax)
    sns.kdeplot(data=df_filtered, x="PPPPC", hue="Region", hue_order=order_region, multiple="stack", alpha=1, palette=custom, fill=False, linewidth=1, linestyle='-', ax=ax)

    # Configuration grid and labels
    ax.set_title('Distribution of GDP per Capita (PPP) by Region', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('GDP per capita (PPP)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_xlim(0, max_value)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x * 100000):,}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Median line
    ax.axvline(mean_value, color='darkred', linestyle='--', linewidth=1, label=f'Median: {mean_value:.2f}')
    ax.text(
        x=mean_value + (max_value * 0.01),
        y=ax.get_ylim()[1] * 0.99,
        s=f'Median: {mean_value:,.0f}',
        color='darkred',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        weight='bold')

    # Añadir la leyenda al gráfico
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=region, alpha=0.4) for region, color in custom.items()]
    legend = ax.legend(handles=legend_elements, title='Region', title_fontsize='10', fontsize='9', loc='upper right')
    plt.setp(legend.get_title(), fontweight='bold')

    # Add Year label
    ax.text(0.95, 1.06, f'{year}',
        transform=ax.transAxes,
        fontsize=22, ha='right', va='top',
        fontweight='bold', color='#D3D3D3')

# Configurate animation
years = sorted(df['Year'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=250, blit=False)

# Save the animation :)
ani.save('C:/Users/guill/Downloads/FIG_GDP_Capita_PPP_KDEPLOT.webp', writer='imagemagick', fps=3)

# Print it!
plt.show()