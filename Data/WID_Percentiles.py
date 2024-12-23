# Libraries
# ===================================================
import requests
import os
import pandas as pd
import numpy as np

# Data Extraction (Countries)
# =====================================================================
# Extract JSON and bring data to a dataframe
url = 'https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json'
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df_countries = df.rename(columns={'index': 'ISO3'})

# Data Extraction
# ===================================================
# Define CSV path
path = r'C:\Users\guill\Downloads\data'

# List to save dataframe
list = []

# Iterate over each file
for archivo in os.listdir(path):
    if archivo.startswith("WID_data_") and archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, archivo), delimiter=';')
        list.append(df)

# Combine all dataframe
dfx = pd.concat(list, ignore_index=True)

# Data Manipulation
# ===================================================
# Filter dataframe
variable = ['sdiincj992', 'shwealj992']
df = dfx[dfx['variable'].isin(variable)]

# Clean dataframe
df = df[~df['percentile'].str.contains(r'\.', na=False)]
df['dif'] = df['percentile'].str.extract(r'p(\d+)p(\d+)').astype(int).apply(lambda x: x[1] - x[0], axis=1)
df = df[df['dif'] == 1]
df['percentile'] = df['percentile'].str.extract(r'p\d+p(\d+)').astype(int)
df = df.sort_values(by=['country', 'variable', 'year', 'percentile'])
df['value'] =  df['value'] * 100

# Pivot dataframe
df['variable'] = df['variable'].replace({'sdiincj992': 'income', 'shwealj992': 'wealth'}) 
df = df[['variable', 'country', 'year', 'percentile', 'value']]
df = df.pivot_table(index=['country', 'year', 'percentile'], columns='variable', values='value')
df = df.reset_index()

# Expand data from 1950
countries = df['country'].unique()
years = np.arange(1950, 2022)
percentiles = df['percentile'].unique()
dfexp = pd.MultiIndex.from_product([countries, years, percentiles], names=['country', 'year', 'percentile']).to_frame(index=False)
df = pd.merge(dfexp, df, on=['country', 'year', 'percentile'], how='left')

# DF - First Year Wealth
dfw = df[df['wealth'].notna()]
dfw = dfw.groupby(['country'], as_index=False)['year'].min()
dfw = pd.merge(df, dfw, on=['country', 'year'], how='inner')
dfw = dfw[['country', 'percentile', 'wealth']]
dfw = dfw.rename(columns={'wealth': 'wealth_null'})

# DF - First Year Income
dfi = df[df['income'].notna()]
dfi = dfi.groupby(['country'], as_index=False)['year'].min()
dfi = pd.merge(df, dfi, on=['country', 'year'], how='inner')
dfi = dfi[['country', 'percentile', 'income']]
dfi = dfi.rename(columns={'income': 'income_null'})

# Replace nulls for first values
df = pd.merge(df, dfw, on=['country', 'percentile'], how='inner')
df['wealth'] = np.where(df['wealth'].isna(), df['wealth_null'], df['wealth'])
df = pd.merge(df, dfi, on=['country', 'percentile'], how='inner')
df['income'] = np.where(df['income'].isna(), df['income_null'], df['income'])
df = df[['country', 'year', 'percentile', 'wealth', 'income']]

df.to_parquet(r'C:\Users\guill\Downloads\data\WID_Percentiles.parquet', engine='pyarrow')
