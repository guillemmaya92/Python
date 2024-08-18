# Libraries
# ==================================================
import requests
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from ecbdata import ecbdata

# Getting Currencies
# ==================================================
# Building blocks for the URL
entrypoint = 'https://sdw-wsrest.ecb.europa.eu/service/' # Using protocol 'https'
resource = 'data'           # The resource for data queries is always'data'
flowRef ='EXR'              # Dataflow describing the data that needs to be returned, exchange rates in this case
key = 'M..EUR.SP00.A'    # Defining the dimension values, explained below

# Define the parameters
parameters = {
    'startPeriod': '2000-01-01',  # Start date of the time series
    'endPeriod': '2024-12-31'     # End of the time series
}

# Construct the URL: https://sdw-wsrest.ecb.europa.eu/service/data/EXR/D.CHF.EUR.SP00.A
request_url = entrypoint + resource + '/'+ flowRef + '/' + key

# Make the HTTP request
response = requests.get(request_url, params=parameters, headers={'Accept': 'text/csv'})
response.text[0:1000]
df_exr = pd.read_csv(io.StringIO(response.text))
df_exr['TIME_PERIOD'] = pd.to_datetime(df_exr['TIME_PERIOD'])
df_exr['OBS_VALUE'] = df_exr['OBS_VALUE'].astype(float)
df_exr = df_exr[['TIME_PERIOD', 'OBS_VALUE', 'UNIT']]
df_exr.columns = ['Date', 'Exchange', 'Unit']

# Make a copy to get USA
df_exr_usd = df_exr.copy()
df_exr_usd = df_exr_usd[df_exr_usd['Unit'] == 'USD']
df_exr_usd.columns = ['Date', 'Exchange_USD', 'Unit_USD']

# Getting M2
# ==================================================
# Define M2 codes and empty list
list_m2 = ['BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E', 'RTD.M.US.Y.M_M2.U', 'RTD.M.JP.Y.M_M2.J']
df_list = []

# Iterate over each code in list
for series_code in list_m2:
    df = ecbdata.get_series(series_code, start='1980-01')
    df_list.append(df)

# Concatenate dataframes in one
df_m2 = pd.concat(df_list, ignore_index=True)

# Format and rename columns
df_m2['TIME_PERIOD'] = pd.to_datetime(df_m2['TIME_PERIOD'], format='%Y-%m')
df_m2['OBS_VALUE'] = df_m2['OBS_VALUE'].astype(float)
df_m2['OBS_VALUE'] = np.where(df_m2['UNIT'] == "EUR", df_m2['OBS_VALUE'] / 1000, df_m2['OBS_VALUE'])
df_m2 = df_m2[['TIME_PERIOD', 'OBS_VALUE', 'UNIT']]
df_m2.columns = ['Date', 'M2', 'Unit']

# Getting GDP and Countries
# =====================================================================
# Extract JSON and bring data to a dataframe
url = 'https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json'
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df_countries = df.rename(columns={'index': 'ISO3'})

#Parametro
parameters = ['NGDPD']

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
df_imf['Date'] = pd.to_datetime(df_imf['Year'].astype(str) + '-01-01')

# Merge dataframes and select columns
df_imf = df_imf.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df_imf = df_imf[['Date', 'Cod_Currency', 'Value']]
df_imf.columns = ['Date', 'Unit', 'GDP']

# Filter currencies and grouping countries
df_imf = df_imf[df_imf['Unit'].isin(['EUR', 'USD', 'JPY'])]
df_imf = df_imf.groupby(['Date', 'Unit'])['GDP'].sum().reset_index()

# Expand from yearly to monthly
df_month = pd.DataFrame({'num': list(range(1, 13))})
df_imf = pd.merge(df_imf, df_month, how='cross')
df_imf['Date'] = df_imf.apply(lambda row: row['Date'] + pd.DateOffset(months=row['num'] -1), axis=1)

# Manipulation Data
# ==================================================
# Merge dataframes and convert to EUR
df = pd.merge(df_m2, df_exr, how='left', on=['Date', 'Unit'])
df = pd.merge(df, df_exr_usd, how='left', on=['Date'])
df['M2_USD'] = np.where(df['Unit'] == 'EUR', df['M2'] * df['Exchange_USD'], df['M2'] / df['Exchange'] * df['Exchange_USD'])

# Merge dataframes to add GDP
df = pd.merge(df, df_imf, how='left', on=['Date', 'Unit'])
df['M2_GDP'] = df['M2_USD'] / df['GDP']

# Filter range dates
df = df[(df['Date'] > '2004-12-31') & (df['Date'] < '2024-01-01')]

# Filter last dates
df['Year'] = df['Date'].dt.year
df = df.sort_values(by=['Year', 'Unit', 'Date'])
df = df.groupby(['Year', 'Unit']).tail(1)

# Calculate cummulative variations
first_values = df.groupby('Unit')['M2'].transform('first')
first_values = df.groupby('Unit')['GDP'].transform('first')
first_values = df.groupby('Unit')['M2_GDP'].transform('first')
df['M2_cumvar'] = (df['M2'] - first_values) / first_values
df['GDP_cumvar'] = (df['GDP'] - first_values) / first_values
df['M2_GDP_cumvar'] = (df['M2_GDP'] - first_values) / first_values

# Visualization Data
# ==================================================
# Crear el gráfico
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Date', y='M2_GDP_cumvar', hue='Unit', marker='o')

# Configurar etiquetas y título
plt.xlabel('Date')
plt.ylabel('M2_cumvar')
plt.title('M2_cumvar over Time')
plt.legend(title='Unit')

# Mostrar el gráfico
plt.show()