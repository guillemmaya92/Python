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

# Data Extraction - WBD (1960-1980)
# ========================================================
# To use the built-in plotting method
indicator = ['NY.GDP.PCAP.CD', 'SP.POP.TOTL']
countries = df_countries['ISO3'].tolist()
data_range = range(1960, 2024)
data = wb.data.DataFrame(indicator, countries, data_range, numericTimeKeys=True, labels=False, columns='series').reset_index()
df_wb = data.rename(columns={
    'economy': 'ISO3',
    'time': 'Year',
    'SP.POP.TOTL': 'LP',
    'NY.GDP.PCAP.PP.CD': 'PPPPC'
})

# Adjust LP and filter before 1980
df_wb['LP'] = df_wb['LP'] / 1000000
df_wb = df_wb[df_wb['Year'] < 1980]

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

# Pivot Parameter to columns and filter nulls
df_imf = df_imf.pivot(index=['ISO3', 'Year'], columns='Parameter', values='Value').reset_index()

# Filter after 2024
df_imf = df_imf[df_imf['Year'] >= 1980]

# Data Manipulation
# =====================================================================
# Concat and filter dataframes
df = pd.concat([df_wb, df_imf], ignore_index=True)
df = df.dropna(subset=['PPPPC', 'LP'], how='any')

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

# Function to create a new distribution
def distribution(df):
    average = df['PPPPC'].mean()
    inequality = np.geomspace(1, 10, len(df))
    df['PPPPC_Dis'] = inequality * (average / np.mean(inequality))
    
    return df

df = df.groupby(['Country', 'Year']).apply(distribution).reset_index(drop=True)

# Logarithmic distribution
df['PPPPC_Dis_Log'] = np.log(df['PPPPC_Dis'])

# Logarithmic distribution
df['Region'] = np.where(df['ISO3'] == 'CHN', 'China', df['Region'])
df['Region'] = np.where(df['ISO3'] == 'USA', 'USA', df['Region'])

print(df)

# Data Visualization
# =====================================================================
# Seaborn figure style
sns.set(style="whitegrid")

# Create a palette
fig, ax = plt.subplots(figsize=(16, 9))

def update(year):
    ax.clear()
    df_filtered = df[df['Year'] == year]
    
    # Calculate mean value
    min_value = df_filtered['PPPPC_Dis_Log'].min()
    max_value = df_filtered['PPPPC_Dis_Log'].max()
    mean_value = df_filtered['PPPPC_Dis_Log'].median()
    mean_value_r = df_filtered['PPPPC_Dis'].median()
    per10 = df_filtered['PPPPC_Dis_Log'].quantile(0.001)
    per90 = df_filtered['PPPPC_Dis_Log'].quantile(0.999)
    population = len(df_filtered)
    
    # Custom palette area
    custom_area = {
        'China': '#e3d6b1',
        'Asia': '#fff3d0',
        'Europe': '#ccdccd',
        'Oceania': '#90a8b7',
        'USA': '#f09c9c',
        'Americas': '#fdcccc',
        'Africa': '#ffe3ce'
    }
 
    # Custom palette line
    custom_line = {
        'China': '#cc9d0e',
        'Asia': '#FFC107',
        'Europe': '#004d00',
        'Oceania': '#003366',
        'USA': '#a60707',
        'Americas': '#FF0000',
        'Africa': '#FF6F00'
    }
    
    # Region Order
    order_region = ['China', 'Asia', 'Africa', 'USA', 'Americas', 'Europe', 'Oceania'] 

    # Create kdeplot area and lines
    sns.kdeplot(data=df_filtered, x="PPPPC_Dis_Log", hue="Region", bw_adjust=2, hue_order=order_region, multiple="stack", alpha=1, palette=custom_area, fill=True, linewidth=1, linestyle='-', ax=ax)
    sns.kdeplot(data=df_filtered, x="PPPPC_Dis_Log", hue="Region", bw_adjust=2, hue_order=order_region, multiple="stack", alpha=1, palette=custom_line, fill=False, linewidth=1, linestyle='-', ax=ax)

    # Configuration grid and labels
    fig.suptitle('Global GDP per Capita distribution', fontsize=16, fontweight='bold', y=0.95)
    ax.set_title('Evolution by region from 1980 to 2030', fontsize=12, fontweight='normal', pad=18)
    ax.set_xlabel('GDP per capita (PPP), log axis', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency of total population (M)', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    ax.set_ylim(0, 0.4)
    ax.set_xlim(per10, per90)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x * 100000):,}'))
    
    # Inverse logarhitmic xticklabels
    xticks = np.linspace(df_filtered["PPPPC_Dis_Log"].min(), df_filtered["PPPPC_Dis_Log"].max(), num=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(np.exp(tick)):,}' for tick in xticks])
    
    # White color to xticklabels
    for label in ax.get_xticklabels():
        label.set_color('black')
        
    # Median line
    ax.axvline(mean_value, color='darkred', linestyle='--', linewidth=0.5)
    ax.text(
        x=mean_value + (max_value * 0.01),
        y=ax.get_ylim()[1] * 0.98,
        s=f'Median: ${mean_value_r:,.0f} ppp',
        color='darkred',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        weight='bold')
    
    # Population label
    ax.text(
        0.02,
        0.98,
        s=f'Population: {population / 10:,.0f} (M)',
        transform=ax.transAxes,
        color='dimgrey',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        weight='bold')
    
    # Add a custom legend
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=region, alpha=0.4) for region, color in custom_line.items()]
    legend = ax.legend(handles=legend_elements, title='Region', title_fontsize='10', fontsize='9', loc='upper right')
    plt.setp(legend.get_title(), fontweight='bold')

    # Add Year label
    ax.text(0.95, 1.06, f'{year}',
        transform=ax.transAxes,
        fontsize=22, ha='right', va='top',
        fontweight='bold', color='#D3D3D3')
    
    # Add label "poorest" and "richest"
    plt.text(0, -0.065, 'Poorest',
             transform=ax.transAxes,
             fontsize=10, fontweight='bold', color='darkred', ha='left', va='center')
    plt.text(0.95, -0.065, 'Richest',
             transform=ax.transAxes,
             fontsize=10, fontweight='bold', color='darkblue', va='center')

    # Add Data Source
    plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
            transform=plt.gca().transAxes, 
            fontsize=8, 
            color='gray')

    # Add Notes
    plt.text(0, -0.12, 'Notes: Notes: The distribution of values, based on GDP per capita, has been calculated using a logarithmic scale ranging from 1 to 10 and adjusted proportionally to the population size of each country.', 
        transform=plt.gca().transAxes,
        fontsize=8, 
        color='gray')

# Configurate animation
years = sorted(df['Year'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=250, blit=False)

# Save the animation :)
ani.save('C:/Users/guill/Downloads/FIG_GDP_Capita_Distribution_PPP_KDEPLOT.mp4', writer='ffmpeg', fps=3)

# Print it!
plt.show()
