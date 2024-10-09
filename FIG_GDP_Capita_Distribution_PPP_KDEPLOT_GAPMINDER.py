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
df_countries = df.rename(columns={'index': 'iso3'})

# Data Extraction (GAPMINDER)
# ====================================================================
# URL Github
urlgdp = 'https://raw.githubusercontent.com/open-numbers/ddf--gapminder--gdp_per_capita_cppp/refs/heads/master/ddf--datapoints--income_per_person_gdppercapita_ppp_inflation_adjusted--by--geo--time.csv'
urlpop = 'https://raw.githubusercontent.com/open-numbers/ddf--gapminder--population_historic/refs/heads/master/ddf--datapoints--population_total--by--geo--time.csv'

# Create dataframes
dfgdp = pd.read_csv(urlgdp)
dfpop = pd.read_csv(urlpop)

# Merge merges and rename dataframes
dfgap = pd.merge(dfgdp, dfpop, on=['geo', 'time'], how='inner')
dfgap.rename(columns={'geo': 'iso3', 'time': 'year', 'income_per_person_gdppercapita_ppp_inflation_adjusted': 'gdpccppp', 'population_total': 'pop'}, inplace=True)

# Transform iso3 to upper and divide population
dfgap['iso3'] = dfgap['iso3'].str.upper()
dfgap['pop'] = dfgap['pop'] // 1000000

# Data Manipulation
# ====================================================================
# Copy Dataframe
df = dfgap.copy()

# Create a list
dfs = []

# Interpolate monthly data
for iso3 in df['iso3'].unique():
    temp_df = df[df['iso3'] == iso3].copy()
    temp_df['date'] = pd.to_datetime(temp_df['year'], format='%Y')
    temp_df = temp_df[['date', 'pop', 'gdpccppp']]
    temp_df = temp_df.set_index('date').resample('ME').mean().interpolate(method='linear').reset_index()
    temp_df['iso3'] = iso3
    temp_df['year'] = temp_df['date'].dt.year 
    dfs.append(temp_df)

# Concat dataframes    
df = pd.concat(dfs, ignore_index=True)

# Merge queries
df = df.merge(df_countries, how='left', left_on='iso3', right_on='iso3')
df = df[['iso3', 'Country', 'Region', 'year', 'date', 'pop', 'gdpccppp']]
df = df[df['Region'].notna()]

# Expand dataframe with population
columns = df.columns
df = np.repeat(df.values, df['pop'].astype(int), axis=0)
df = pd.DataFrame(df, columns=columns)

# Function to create a new distribution
def distribution(df):
    average = df['gdpccppp'].mean()
    inequality = np.geomspace(1, 10, len(df))
    df['gdpccpppd'] = inequality * (average / np.mean(inequality))
    
    return df

df = df.groupby(['iso3', 'year', 'date']).apply(distribution).reset_index(drop=True)

# Logarithmic distribution
df['gdpccpppdl'] = np.log(df['gdpccpppd'])

# Logarithmic distribution
df['Region'] = np.where(df['iso3'] == 'CHN', 'China', df['Region'])
df['Region'] = np.where(df['iso3'] == 'USA', 'USA', df['Region'])

print(df)

# Data Visualization
# =====================================================================
# Seaborn figure style
sns.set(style="whitegrid")

# Create a palette
fig, ax = plt.subplots(figsize=(16, 9))

def update(year):
    ax.clear()
    df_filtered = df[df['date'] == year]
    
    # Calculate mean value
    min_value = df_filtered['gdpccpppdl'].min()
    max_value = df_filtered['gdpccpppdl'].max()
    mean_value = df_filtered['gdpccpppdl'].median()
    mean_value_r = df_filtered['gdpccpppd'].median()
    per10 = df_filtered['gdpccpppdl'].quantile(0.001)
    per90 = df_filtered['gdpccpppdl'].quantile(0.999)
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
    sns.kdeplot(data=df_filtered, x="gdpccpppdl", hue="Region", bw_adjust=2, hue_order=order_region, multiple="stack", alpha=1, palette=custom_area, fill=True, linewidth=1, linestyle='-', ax=ax)
    sns.kdeplot(data=df_filtered, x="gdpccpppdl", hue="Region", bw_adjust=2, hue_order=order_region, multiple="stack", alpha=1, palette=custom_line, fill=False, linewidth=1, linestyle='-', ax=ax)

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
    xticks = np.linspace(df_filtered["gdpccpppdl"].min(), df_filtered["gdpccpppdl"].max(), num=5)
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
    formatted_date = year.strftime('%Y-%m') 
    ax.text(0.95, 1.06, f'{formatted_date}',
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
years = sorted(df['date'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=1000, blit=False)

# Save the animation :)
ani.save('C:/Users/guillem.maya/Downloads/FIG_GDP_Capita_Distribution_PPP_KDEPLOT_GAPMINDER.mp4', writer='ffmpeg', fps=40)

# Print it!
plt.show()