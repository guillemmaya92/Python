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
from matplotlib.lines import Line2D
from datetime import datetime

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
indicator = ['NY.GDP.MKTP.KD', 'NY.GDP.MKTP.CN', 'NY.GDP.MKTP.CD', 'SP.POP.TOTL']
countries = df_countries['ISO3'].tolist()
data_range = range(1960, 2024)
data = wb.data.DataFrame(indicator, countries, data_range, numericTimeKeys=True, labels=False, columns='series').reset_index()
df_wb = data.rename(columns={
    'economy': 'ISO3',
    'time': 'Year',
    'SP.POP.TOTL': 'LP',
    'NY.GDP.MKTP.KD': 'GDPCNT',
    'NY.GDP.MKTP.CN': 'GDPLOC',
    'NY.GDP.MKTP.CD': 'GDPUSD'
})

# Adjust LP and filter before 1980
df_wb['LP'] = df_wb['LP'] / 1000000
df_wb = df_wb[df_wb['Year'] >= 1990]

df_wb['GDPUSD_Inc'] = (df_wb['GDPUSD'] / df_wb.groupby('ISO3')['GDPUSD'].shift(1)) -1
df_wb['GDPLOC_Inc'] = (df_wb['GDPLOC'] / df_wb.groupby('ISO3')['GDPLOC'].shift(1)) -1
df_wb['GDPCNT_Inc'] = (df_wb['GDPCNT'] / df_wb.groupby('ISO3')['GDPCNT'].shift(1)) -1
df_wb['DIF_Total'] = df_wb['GDPCNT_Inc'] - df_wb['GDPUSD_Inc']
df_wb['DIF_Currency'] = df_wb['GDPLOC_Inc'] - df_wb['GDPUSD_Inc']
df_wb['DIF_Inflation'] = df_wb['GDPCNT_Inc'] - df_wb['GDPLOC_Inc']

df_wb = df_wb[df_wb['ISO3'] == 'ESP']

print(df_wb)