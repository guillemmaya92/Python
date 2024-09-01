# Libraries
# =====================================================================
import requests
import wbgapi as wb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Data Extraction - GITHUB (Countries)
# =====================================================================
# Extract JSON and bring data to a dataframe
url = 'https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json'
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df_countries = df.rename(columns={'index': 'ISO3'})

# Data Extraction - WBD (1960-2024)
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

# Adjust LP and filter before 2024
df_wb['LP'] = df_wb['LP'] / 1000000
df_wb = df_wb[df_wb['Year'] < 1980]

# Data Extraction - IMF (2024-2030)
# =====================================================================
#Parametro
parameters = ['NGDPDPC', 'LP']

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

# Merge and filter dataframes
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[df['Region'].notna()]
df = df[['Region', 'Cod_Currency', 'Country', 'ISO3', 'Year', 'NGDPDPC', 'LP']]

# Add Country column
df['ISO3'] = df.apply(lambda row: 'EUR' if row['Cod_Currency'] == 'EUR' else row['ISO3'], axis=1)
df['Country'] = df.apply(lambda row: 'Europe' if row['Cod_Currency'] == 'EUR' else row['Country'], axis=1)

# Add GDP and  % of GDP Share
df['GDP'] = df['NGDPDPC'] * df['LP']
df = df.groupby(['ISO3', 'Country', 'Year'], as_index=False).agg({'GDP': 'sum'})
df['GDP_Share'] = df['GDP'] / df.groupby('Year')['GDP'].transform('sum')

# Filter Countries
ISO3 = ['EUR', 'GBR', 'USA', 'CHN', 'JPN', 'IND']
df = df[df['ISO3'].isin(ISO3)]

# Select columns
df = df[['ISO3', 'Country', 'Year', 'GDP', 'GDP_Share']]

# Create real and prediction dataframe
dfr = df.copy()
dfr = dfr[dfr['Year'] <= 2023]
dfp = df.copy()
dfp = dfp[dfp['Year'] >= 2023]

# Data Visualization
# =====================================================================
# Seaborn style and font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Line colors
palette = {
    'EUR': '#4EA72E',
    'GBR': '#A02B93',
    'USA': '#0B3040',
    'CHN': '#C00000',
    'JPN': '#808080',
    'IND': '#E97132',
}

country = {
    'EUR': 'Eurozone',
    'GBR': 'United Kingdom',
    'USA': 'United States',
    'CHN': 'China',
    'JPN': 'Japan',
    'IND': 'India',
}

# Create the figure and lines
plt.figure(figsize=(16, 9))
sns.lineplot(data=dfr, x='Year', y='GDP_Share', hue='ISO3', palette=palette, linestyle='-',  legend='auto', alpha=0.7)
sns.lineplot(data=dfp, x='Year', y='GDP_Share', hue='ISO3', palette=palette, linestyle='--', legend=False, alpha=0.7)

# Title and labels
plt.suptitle("     Share of World's GDP", fontsize=16, fontweight='bold', y=0.95)
plt.title('Historical Evolution from 1980 to 2023 and Projections through 2029', fontsize=12, fontweight='bold', color='darkgrey', pad=15)
plt.xlabel('Year', fontsize=10, fontweight='bold')
plt.ylabel('Share of GDP (%)', fontsize=10, fontweight='bold')

# Configuration
plt.grid(True, which='major', axis='x', linestyle=':', color='grey', linewidth=0.3)
plt.xlim(1960, 2029)

# Add Data Source
plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Create a custom legend
handles, labels = plt.gca().get_legend_handles_labels()
labels = [country[label] for label in labels]
plt.legend(handles, labels, title='Country', title_fontsize='10', fontsize='9')

legend = plt.gca().get_legend()
if legend:
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize('10')
    
# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_GDP_Share_Evolution.png', format='png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()