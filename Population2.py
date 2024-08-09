# Libraries
# =====================================================================
import requests
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
df = df[(df['ISO3'] == 'ESP') & (df['Year'] == 2023)]

# Expand with a population distribution
columns = df.columns
df = np.repeat(df.values, df['LP'].astype(int), axis=0)
df = pd.DataFrame(df, columns=columns)
        
def calcular_columnas(df):
  
    # Calcula las columnas necesarias
    dfc = df.copy()
    dfc['Recuento'] = dfc.index + 1
    dfc['Acumulativo'] = dfc['Recuento'] + dfc['Recuento'].shift(1).fillna(0)
    dfc['Suma'] = dfc['Acumulativo'].sum()
    dfc['Division'] = dfc['Acumulativo'] / dfc['Suma']
    dfc['Multiplicacion'] = dfc['PPPPC'] * dfc['Division']
    dfc['Prorrateado'] = dfc['Multiplicacion'] * len(dfc)
    df['PPPPC'] = dfc['Prorrateado']
    return df

df = df.groupby(['Country', 'Year']).apply(calcular_columnas).reset_index(drop=True)

print(df)