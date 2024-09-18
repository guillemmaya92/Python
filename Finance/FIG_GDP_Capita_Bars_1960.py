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
    'NY.GDP.PCAP.CD': 'NGDPDPC'
})

# Adjust LP and filter before 1980
df_wb['LP'] = df_wb['LP'] / 1000000
df_wb = df_wb[df_wb['Year'] < 1980]

# Data Extraction - IMF (1980-2030)
# =====================================================================
#Parametro
parameters = ['LP', 'NGDPDPC']

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
df = df.dropna(subset=['NGDPDPC', 'LP'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['ISO3', 'Country', 'Year', 'LP', 'NGDPDPC', 'Analytical', 'Region']]
df = df[df['Region'].notna()]

# Filter nulls and order
df = df.sort_values(by=['Year', 'NGDPDPC'])

# Calculate 'left accrual widths'
df['LPcum'] = df.groupby('Year')['LP'].cumsum()
df['Left'] = df.groupby('Year')['LP'].cumsum() - df['LP']

# Calculate GDP Average weighted by Population and partitioned by Year
df['AVG_Weight'] = df.groupby('Year')['NGDPDPC'].transform(lambda x: np.average(x, weights=df.loc[x.index, 'LP']))

# Add a total GDP column and cummulative it
df['GDP'] = df['NGDPDPC'] * df['LP']
df['GDPcum'] = df.groupby('Year')['GDP'].cumsum()
df['NGDPDPC_Change'] = ((df['NGDPDPC'] / df.groupby('ISO3')['NGDPDPC'].transform('first')) - 1) * 100

# Define function to calculate Gini coefficient
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

df['Gini'] = df.groupby('Year')['NGDPDPC'].transform(lambda x: gini(x))

# Define function to calculate a variation coefFicient
def variation(x):
    x = np.array(x)
    mean = np.mean(x)
    standard_dev = np.std(x, ddof=0)
    var = (standard_dev / mean) * 100
    return var

df['Variation'] = df.groupby('Year')['NGDPDPC'].transform(lambda x: variation(x))

# Copy a df sample to calculate a median
df_sample = df.copy()
columns = df.columns
df_sample = np.repeat(df_sample.values, df_sample['LP'].astype(int) * 10, axis=0)
df_sample = pd.DataFrame(df_sample, columns=columns)
df_sample = df_sample.groupby('Year')['NGDPDPC'].median().reset_index()
df_sample = df_sample.rename(columns={'NGDPDPC': 'Median'})
df_sample['Median_Change'] =  ((df_sample['Median'] /  df_sample['Median'].iloc[0]) -1) * 100

# Merge queries
df = df.merge(df_sample, how='left', on='Year')

# Data Visualization
# =====================================================================
# Seaborn figure style
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(16, 9))

# Create a palette
palette = sns.color_palette("coolwarm", as_cmap=True).reversed()

# Function to refresh animation
def update(year):
    plt.clf()
    subset = df[df['Year'] == year]
    subset_usa = subset[subset['ISO3'] == 'USA'].copy()
    subset_top = subset[subset['ISO3'].isin(['USA', 'DEU', 'JPN', 'GBR'])].copy()
    
    # Normalize GDPcum in a range [0, 1]
    gdp_min = subset['GDPcum'].min()
    gdp_max = subset_usa['GDPcum'].max()
    norm = plt.Normalize(gdp_min, gdp_max)
    colors = palette(norm(subset['GDPcum']))
    
    # Create a Matplotlib plot
    bars = plt.bar(subset['Left'], subset['NGDPDPC'], width=subset['LP'], 
            color=colors, alpha=1, align='edge', edgecolor='grey', linewidth=0.1)
    
    # Configuration grid and labels
    plt.xlim(0, subset['LPcum'].max())
    plt.ylim(0, subset_top['NGDPDPC'].max() * 1.05)
    plt.grid(axis='x')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    plt.title(f'Global Distribution of GDP by Country', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cumulative Global Population (M)', fontsize=10, fontweight='bold')
    plt.ylabel('GDP per capita (US$)', fontsize=10, fontweight='bold')
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9) 
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        
    # Add Labels to relevant countries
    for bar, value, country in zip(bars, subset['GDP'], subset['ISO3']):
        if country in ['CHN', 'IND', 'USA', 'IDN', 'PAK', 'NGA', 'BRA', 'BGD', 'RUS', 'MEX', 'JPN', 'VNM', 'DEU', 'GBR']:
            plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{country}\n{''}', ha='center', va='bottom', fontsize=7, color='grey')

    # Add Median Line and Label
    median = subset['Median'].max()
    median_change = subset.iloc[0]['Median_Change']
    maxis = subset_usa['NGDPDPC'].max()
    
    plt.axhline(
        y=median,
        color='darkred', 
        linestyle='--', 
        linewidth=0.5)

    plt.text(
        x=subset['Left'].max() * 0.02,
        y=median + (maxis * 0.04),
        s=f'GDP Capita Median: {median:,.0f}',
        color='darkred',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        weight='bold')

    plt.gca().text(
                subset['Left'].max() * 0.02,
                median + (maxis * 0.02),
                f'Cumulative growth: {median_change:,.0f}%', 
                ha='left', va='center', 
                fontsize=9, 
                color='darkgreen')
    
    # Add USA Line and Label
    pibc_usa = subset_usa.iloc[0]['NGDPDPC']
    pibc_usa_change = subset_usa.iloc[0]['NGDPDPC_Change']
    
    plt.axhline(
        y=pibc_usa, 
        color='darkblue', 
        linestyle='--', 
        linewidth=0.5)
    
    plt.text(
        x=subset['Left'].max() * 0.02,
        y=pibc_usa * 0.95,
        s=f'GDP Capita USA: {pibc_usa:,.0f}',
        color='darkblue',
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='left',

        weight='bold')

    plt.gca().text(
                subset['Left'].max() * 0.02,
                pibc_usa * 0.93,
                f'Cumulative growth: {pibc_usa_change:,.0f}%', 
                ha='left', va='center', 
                fontsize=9, 
                color='darkgreen')
    
    plt.gca().text(
                subset['Left'].max() * 0.02,
                pibc_usa * 0.9,
                f'Median-relative: {pibc_usa / median * 100:,.0f}%', 
                ha='left', va='center', 
                fontsize=9, 
                color='darkgrey')

    # Add Year label 
    plt.text(0.95, 1.06, f'{year}',
             transform=plt.gca().transAxes,
             fontsize=22, ha='right', va='top',
             fontweight='bold', color='#D3D3D3')
    
    # Add Data Source
    plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024 | World Bank national accounts data, and OECD National Accounts data files.', 
            transform=plt.gca().transAxes, 
            fontsize=8, 
            color='gray')

    # Add label "poorest" and "richest"
    plt.text(0, -0.065, 'Poorest',
             transform=ax.transAxes,
             fontsize=12, fontweight='bold', color='darkred', ha='left', va='center')
    plt.text(0.95, -0.065, 'Richest',
             transform=ax.transAxes,
             fontsize=12, fontweight='bold', color='darkblue', va='center')

# Configurate animation
years = sorted(df['Year'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=250, blit=False)

# Save the animation :)
ani.save('C:/Users/guill/Downloads/FIG_GDP_Capita_Bars_1960.mp4', writer='ffmpeg', fps=2.5)

# Print it!
plt.show()
