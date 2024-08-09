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

# Expand with a new distribution
dfgc = pd.DataFrame({'Values': [0.22, 0.43, 0.87, 1.3, 2.17]})
df = df.merge(dfgc, how='cross')
df['PPPPC'] = df['PPPPC'] * df['Values']

# Expand with a population distribution
columns = df.columns
df = np.repeat(df.values, df['LP'].astype(int), axis=0)
df = pd.DataFrame(df, columns=columns)

# Filtering years
years = [1980, 1990, 2000, 2010, 2020, 2030]
df = df[df['Year'].isin(years)]

# Logarithmic PPPPC
df['PPPPC'] = pd.to_numeric(df['PPPPC'], errors='coerce')
df['Log_PPPPC'] = np.log(df['PPPPC'])


# Plotting
# =====================================================================
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Log_PPPPC", hue="Year", fill=True)

plt.xlabel('Log_PPPPC')
plt.ylabel('Density')
plt.title('Distribution of Log_PPPPC by Year')
plt.grid(True)

# La leyenda debe ser automáticamente añadida por seaborn si 'hue' está correctamente especificado.
plt.show()