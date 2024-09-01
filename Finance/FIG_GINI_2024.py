# Libraries
# =====================================================================
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Data Manipulation
# =====================================================================
# Pivot Parameter to columns and filter nulls
df = df_imf.pivot(index=['ISO3', 'Year'], columns='Parameter', values='Value').reset_index()
df = df.dropna(subset=['LP', 'NGDPDPC'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['ISO3', 'Country', 'Year', 'LP', 'NGDPDPC', 'Analytical', 'Region']]
df = df[df['Region'].notna()]

# Filter nulls and order
df = df.sort_values(by=['Year', 'NGDPDPC'])

# Copy a df sample to calculate a median
columns = df.columns
df = np.repeat(df.values, df['LP'].astype(int), axis=0)
df = pd.DataFrame(df, columns=columns)

# Function to create a new distribution
def distribution(df):
    average = df['NGDPDPC'].mean()
    inequality = np.geomspace(1, 10, len(df))
    df['NGDPDPC_Dis'] = inequality * (average / np.mean(inequality))
    
    return df

df = df.groupby(['Country', 'Year']).apply(distribution).reset_index(drop=True)

# Function to calculate Global GINI
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

# DataFrame Between-Country
# =====================================================================
# Copy Dataframe
dfb = df.copy()

# Sorting and formatting
dfb = dfb[dfb['Year'] == 2024]
dfb = dfb.sort_values(by=['Year', 'NGDPDPC_Dis'])

# Calculating cummulative population and distribution
dfb['GDP_Cum'] = dfb.groupby('Year')['NGDPDPC_Dis'].cumsum() / dfb.groupby('Year')['NGDPDPC_Dis'].transform('sum')
dfb['POP'] = range(1, len(dfb) + 1)
dfb['POP_Cum'] = dfb['POP'] / len(dfb)

# World Mean and Median
dfb['Mean']  = dfb['NGDPDPC_Dis'].mean()
dfb['Median']  = dfb['NGDPDPC_Dis'].median()

# Calculation Gini
dfb['Gini'] = dfb.groupby('Year')['NGDPDPC_Dis'].transform(lambda x: gini(x))

# Selecting columns and filtering year
dfb = dfb[['ISO3', 'Country', 'Year', 'Gini', 'GDP_Cum', 'POP_Cum', 'NGDPDPC_Dis', 'Mean', 'Median']]

# DataFrame Within-Country
# =====================================================================
# Copy Dataframe
dfw = df.copy()

# Sorting and formatting
dfw = dfw[(dfw['ISO3'] == 'USA') & (dfw['Year'] == 2024)]
dfw = dfw.sort_values(by=['Year', 'NGDPDPC_Dis'])
dfw['LP'] = pd.to_numeric(dfw['LP'], errors='coerce')

# Calculating cummulative population and distribution
dfw['GDP_Cum'] = dfw.groupby('Year')['NGDPDPC_Dis'].cumsum() / dfw.groupby('Year')['NGDPDPC_Dis'].transform('sum')
dfw['POP'] = range(1, len(dfw) + 1)
dfw['POP_Cum'] = dfw['POP'] / len(dfw)

# Calculation Gini
dfw['Gini'] = dfw.groupby('Year')['NGDPDPC_Dis'].transform(lambda x: gini(x))

# Selecting columns and filtering year
dfw = dfw[['ISO3', 'Country', 'Year', 'Gini', 'GDP_Cum', 'POP_Cum']]

# DataFrame Equality
# =====================================================================
data = {
    'POP_Cum': [0, 1],
    'GDP_Cum': [0, 1]
}

dfe = pd.DataFrame(data)

# DataFrame Row Mean and Median
# =====================================================================
dfmean = dfb.loc[[ (dfb['NGDPDPC_Dis'] - dfb['Mean']).abs().idxmin() ]]
dfmedianpop = dfb.loc[[ (dfb['POP_Cum'] - 0.5).abs().idxmin() ]]
dfmedianinc = dfb.loc[[ (dfb['GDP_Cum'] - 0.5).abs().idxmin() ]]

# Data Visualization
# =====================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Create the figure and lines
plt.figure(figsize=(10, 10))
plt.plot(dfw['POP_Cum'], dfw['GDP_Cum'], label='Gini Within-Countries', color='darkblue', linewidth=1, linestyle=':')
plt.plot(dfb['POP_Cum'], dfb['GDP_Cum'], label='Gini Between-Countries', color='darkred')
plt.plot(dfe['POP_Cum'], dfe['GDP_Cum'], label='Perfect Distribution', color='darkgrey')

# Add a horizontal line for mean
ymean = dfmean['GDP_Cum'].iloc[0]
plt.axhline(y=ymean, color='darkgrey', linestyle='--', linewidth=1, label='Mean Income')

# Add scatter median population
xpop = dfmedianpop['POP_Cum'].iloc[0]
ypop = dfmedianpop['GDP_Cum'].iloc[0]
vpop = dfmedianpop['NGDPDPC_Dis'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Population', zorder=5)
plt.text(x=xpop+0.07, y=ypop-0.02, 
        s=f'Median Population:\n{vpop: ,.0f}$\n{(xpop) * 100: ,.0f}% - {(ypop) * 100: ,.1f}%', 
        color='darkred', 
        va='center', 
        ha='left', 
        fontsize=8)

# Add scatter median income
xpop = dfmedianinc['POP_Cum'].iloc[0]
ypop = dfmedianinc['GDP_Cum'].iloc[0]
vpop = dfmedianinc['NGDPDPC_Dis'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Income', zorder=5)
plt.text(x=xpop-0.02, y=ypop, 
         s=f'Median Income:\n{vpop: ,.0f}$\n{(1-xpop) * 100: ,.1f}% - {(ypop) * 100: ,.0f}%',
         color='darkred', 
         va='center', 
         ha='right', 
         fontsize=8)

# Add scatter mean income
xpop = dfmean['POP_Cum'].iloc[0]
ypop = dfmean['GDP_Cum'].iloc[0]
vpop = dfmean['NGDPDPC_Dis'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='dimgray', label='Mean Income', zorder=5, marker='o', facecolor='none')
plt.text(x=xpop+0.05, y=ypop-0.05, 
        s=f'Mean Income:\n{vpop: ,.0f}$\n{(xpop) * 100: ,.0f}% - {(ypop) * 100: ,.0f}%',  
        color='dimgray', 
        va='center', 
        ha='left', 
        fontsize=8)

# Title and labels
plt.suptitle('   Global Income Inequality', fontsize=16, fontweight='bold', y=0.95)
plt.title('Within-countries vs Between-countries', fontsize=12, fontweight='bold', color='darkgrey', pad=20)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Configuration
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Get Gini Values
Giniw = dfw['Gini'].iloc[-1] 
Ginib = dfb['Gini'].iloc[-1] 

# Add legend
plt.text(0.05, 0.93, f'Gini Between-Countries: {Ginib:.2f}', color='darkred', fontsize=9, fontweight='bold')
plt.text(0.05, 0.96, f'Gini Within-Countries: {Giniw:.2f}', color='darkblue', fontsize=9)
plt.text(0.05, 0.90, 'Perfect Distribution: 0', color='darkgrey', fontsize=9, fontweight='bold')

# Add Year label 
plt.text(1, 1.06, f'2024',
    transform=plt.gca().transAxes,
    fontsize=16, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')
    
# Add Data Source
plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Add Notes
plt.text(0, -0.15, 'Notes: The distribution of values, based on GDP per capita, has been calculated using a logarithmic scale ranging from 1 to 10 and\nadjusted proportionally to the population size of each country.', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_GINI_2024.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
