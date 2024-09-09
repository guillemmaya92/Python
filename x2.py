# Libraries
# ===================================================
import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# Data Extraction
# ===================================================
# Define CSV path
path = r'C:\Users\guillem.maya\Downloads\data'

# List to save dataframe
list = []

# Iterate over each file
for archivo in os.listdir(path):
    if archivo.startswith("WID_data_") and archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, archivo), delimiter=';')
        list.append(df)

# Combine all dataframes and create a copy
df = pd.concat(list, ignore_index=True)
dfv = df.copy()
dfp = df.copy()

# Filter dataframes
country = ['FR']
variable = ['sdiincj992', 'shwealj992']
variablev = ['adiincj992', 'ahwealj992']
variablep = ['npopuli999']
percentile = ['p10p100', 'p20p100', 'p30p100', 'p40p100', 'p50p100', 'p60p100', 'p70p100', 'p80p100', 'p90p100']
percentilev = ['p0p100']
year = [1985, 2022]
df = df[(df['country'].isin(country)) & df['variable'].isin(variable) & df['percentile'].isin(percentile) & df['year'].isin(year)]
dfv = dfv[(dfv['country'].isin(country)) & dfv['variable'].isin(variablev) & dfv['percentile'].isin(percentilev) & dfv['year'].isin(year)]
dfp = dfp[(dfp['country'].isin(country)) & dfp['variable'].isin(variablep) & dfp['percentile'].isin(percentilev) & dfp['year'].isin(year)]

# Transformation 1
df['value'] = 1 - df['value']
df['percentile'] = df['percentile'].str[1:3].astype(int) / 100
df = df[['country', 'variable', 'year', 'percentile', 'value']]

# Selection columns 2
dfv = dfv[['country', 'variable', 'year', 'value']]

# Selections columns 3
dfp['population'] = (dfp['value'] / 1000).astype(int)
dfp = dfp[['country', 'year', 'population']]

# Create Dataframe to add values 0 and 1
dfx = pd.DataFrame(
    [(c, v, y, p, p) for c in country for v in variable for y in year for p in [0, 1]],
    columns=['country', 'variable', 'year', 'percentile', 'value']
)
df = pd.concat([df, dfx], ignore_index=True)
df = df.sort_values(by=['country', 'variable', 'year', 'percentile']).reset_index(drop=True)

# Merge Population
df = df.merge(dfp, on=['country', 'year'], how='left')

# Data Manipulation
# ===================================================
# Crear an interpolate function
def aplicar_interpolacion(sub_df):
    x = sub_df['percentile'].values
    y = sub_df['value'].values
    p = sub_df['population'].values
    interpolator = PchipInterpolator(x, y)
    
    # Generate new points
    x_smooth = np.linspace(min(x), max(x), num=max(p))
    y_smooth = interpolator(x_smooth)
    
    # Return dataframe with interpolate results
    return pd.DataFrame({
        'country': sub_df['country'].iloc[0],
        'variable': sub_df['variable'].iloc[0],
        'year': sub_df['year'].iloc[0],
        'percentile': x_smooth,
        'value': y_smooth
    })

# Apply function to df partitioned by groups
df = df.groupby(['country', 'variable', 'year']).apply(aplicar_interpolacion).reset_index(drop=True)

# Modify variables to income and wealth
df['variable'] = np.where(df['variable'].str.contains('weal', case=False), 'wealth', 'income')
dfv['variable'] = np.where(dfv['variable'].str.contains('weal', case=False), 'wealth', 'income')

# Merge dataframes
df = df.merge(dfv, on=['country', 'variable', 'year'], how='left')
df = df.sort_values(by=['country', 'variable', 'year', 'percentile'])

# Calculate columns
df['percentile_r'] = df['percentile'] - df.groupby(['country', 'variable', 'year'])['percentile'].shift(1).fillna(0)
df['value_xr'] = df['value_x'] - df.groupby(['country', 'variable', 'year'])['value_x'].shift(1).fillna(0)
df['value_yr'] = df['value_xr'] * df['value_y'] * df.groupby(['country', 'variable', 'year']).transform('size')
df['value_yrm'] = df.groupby(['country', 'variable', 'year'])['value_yr'].transform('median')

# Rename and reorder
df = df.rename(columns={
    'country': 'country',
    'variable': 'variable',
    'year': 'year',
    'percentile': 'population_cum_percent',
    'value_x': 'variable_cum_percent',
    'value_y': 'value_mean',
    'percentile_r': 'population_perecent',
    'value_xr': 'variable_percent',
    'value_yr': 'value',
    'value_yrm': 'value_median',
    })

df = df[
    ['country', 'variable', 'year', 
     'population_perecent', 'variable_percent', 
     'population_cum_percent', 'variable_cum_percent', 
     'value', 'value_mean', 'value_median']
]

# Show results
print(df)