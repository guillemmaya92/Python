# Libraries
# =====================================================================
import requests
import wbgapi as wb
import pandas as pd
import numpy as np
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
df_wb = df_wb[df_wb['Year'] < 2024]

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
df_imf = df_imf[df_imf['Year'] >= 2024]

# Data Manipulation
# =====================================================================
# Concat and filter dataframes
df = pd.concat([df_wb, df_imf], ignore_index=True)
df = df.dropna(subset=['NGDPDPC', 'LP'], how='any')

# Merge and filter dataframes
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[df['Region'].notna()]
df = df[['ISO3', 'Year', 'NGDPDPC', 'LP']]

# Expand rows with population
columns = df.columns
df = np.repeat(df.values, df['LP'].astype(int), axis=0)
df = pd.DataFrame(df, columns=columns)

# Function to create a new distribution
def distribution(df):
    average = df['NGDPDPC'].mean()
    inequality = np.geomspace(1, 10, len(df))
    df['NGDPDPC_Dis'] = inequality * (average / np.mean(inequality))
    return df

df = df.groupby(['ISO3', 'Year']).apply(distribution).reset_index(drop=True)

# Function to calculate Global GINI
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

df['Gini'] = df.groupby('Year')['NGDPDPC_Dis'].transform(lambda x: gini(x))

# Select columns, remove duplicates and order by
df = df[['Year', 'Gini']]
df = df.drop_duplicates()
df = df.sort_values(by='Year')

# Create real and prediction dataframe
dfr = df.copy()
dfr = dfr[dfr['Year'] <= 2023]
dfp = df.copy()
dfp = dfp[dfp['Year'] >= 2023]

# Data Visualization
# =====================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Prepare data to smooth
xr = dfr['Year']
yr = dfr['Gini']

# Interpolate and smooth 
xrs = np.linspace(xr.min(), xr.max(), 500)
yrs = make_interp_spline(xr, yr)(xrs)

# Create the figure and lines
plt.figure(figsize=(16, 9))
plt.plot(xrs, yrs, linestyle='-', color='darkblue')
plt.plot(dfp['Year'], dfp['Gini'], linestyle='--', color='darkblue')

# Add title and labels
plt.suptitle('     Global Income Inequality', fontsize=16, fontweight='bold', y=0.95)
plt.title('Gini Index Between-countries', fontsize=12, fontweight='bold', color='darkgrey', pad=15)
plt.xlabel('Year', fontsize=10, fontweight='bold')
plt.ylabel('Gini Index', fontsize=10, fontweight='bold')

# Configuration
plt.grid(True, which='major', axis='x', linestyle=':', color='grey', linewidth=0.3)
plt.xlim(1960, 2029)

# Add Data Source
plt.text(0, -0.08, 'Data Source: IMF World Economic Outlook Database, 2024', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Add Notes
plt.text(0, -0.1, 'Notes: The distribution of values, based on GDP per capita, has been calculated using a logarithmic scale ranging from 1 to 10 and adjusted proportionally to the population size of each country.', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_GINI_Evolution.png', format='png', dpi=300)

# Mostrar el gráfico
plt.show()
