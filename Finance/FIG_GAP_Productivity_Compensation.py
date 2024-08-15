# Libraries
# ========================================================
import pandas as pd
import pandas_datareader as pdr
import wbgapi as wb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

# FRED Data API Extraction
# ========================================================
# Ranges of dates
start = datetime(1940, 1, 1)
end = datetime.now()

# Data M2SL extraction from FRED
df_fred = pdr.get_data_fred(['M2SL', 'OPHNFB', 'COMPRNFB'], start, end)

# Yearlizer data
df_fred.index = pd.to_datetime(df_fred.index)
df_fred = df_fred.resample('YE').last()
df_fred['Year'] = df_fred.index.year
df_fred = df_fred.reset_index()
df_fred = df_fred[['Year', 'M2SL', 'OPHNFB', 'COMPRNFB']]

# World Bank Data API Extraction
# ========================================================
# To use the built-in plotting method
indicator = ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD', 'SP.POP.TOTL', 'FP.CPI.TOTL.ZG']
countries = ['USA']
data_range = range(1960, 2024)
data = wb.data.DataFrame(indicator, countries, data_range, numericTimeKeys=True, labels=False, columns='series').reset_index()
df_wb = data.rename(columns={
    'economy': 'Country',
    'time': 'Year',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'NY.GDP.MKTP.CD': 'GDP',
    'NY.GDP.PCAP.CD': 'GDPC',
    'SP.POP.TOTL': 'Population'
})

# Manipulation Data
# ========================================================
df = df_fred.merge(df_wb, how='left', left_on='Year', right_on='Year')
df = df[df['Year'] >= 1940]
df['M2var'] = (df['M2SL'] - df['M2SL'].iloc[0]) / df['M2SL'].iloc[0]
df['GDPCvar'] = (df['GDPC'] - df['GDPC'].iloc[0]) / df['GDPC'].iloc[0]
df['Productivityvar'] = (df['OPHNFB'] - df['OPHNFB'].iloc[0]) / df['OPHNFB'].iloc[0]
prod73 = df.loc[df['Year'] == 1973, 'OPHNFB'].iloc[0]
df['Productivityvar73'] = (df['OPHNFB'] - prod73) / prod73
df['Compensationvar'] = (df['COMPRNFB'] - df['COMPRNFB'].iloc[0]) / df['COMPRNFB'].iloc[0]
comp73 = df.loc[df['Year'] == 1973, 'COMPRNFB'].iloc[0]
df['Compensationvar73'] = (df['COMPRNFB'] - comp73) / comp73
df['Inflationvar'] = ((df['Inflation'] / 100) + 1).cumprod() -1

print(df)
# Visualization Data
# ========================================================
# Seaborn figure style
sns.set(style="whitegrid")
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})
plt.figure(figsize=(16, 9))

# Create Matplotlib plots
plt.plot(df['Year'], df['Compensationvar'], label='Compensationvar', color='lightblue')
plt.plot(df['Year'], df['Productivityvar'], label='Productivityvar', color='darkblue')

# Add final lines dots
plt.scatter(df['Year'].iloc[-1], df['Compensationvar'].iloc[-1], color='lightblue', s=25, zorder=5)
plt.scatter(df['Year'].iloc[-1], df['Productivityvar'].iloc[-1], color='darkblue', s=25, zorder=5)

# Add vertical line in 1973
plt.axvline(x=1973, color='darkred', linestyle='--', linewidth=0.5)

# Add productivity label 2024
productivity2024 = df.loc[df['Year'] == 2024, 'Productivityvar'].values[0]
plt.text(2025, productivity2024, 
         f'Productivity:\n{productivity2024 * 100:.2f}%', 
         fontsize=10, fontweight='bold', color='darkblue', 
         ha='left', va='center')

# Add compensation label 2024
compensation2024 = df.loc[df['Year'] == 2024, 'Compensationvar'].values[0]
plt.text(2025, compensation2024, 
         f'Compensation:\n{compensation2024 * 100:.2f}%', 
         fontsize=10, fontweight='bold', color='lightblue', 
         ha='left', va='center')

# Add gap label until 1973
compensation1973 = df.loc[df['Year'] == 1973, 'Compensationvar'].values[0]
productivity1973 = df.loc[df['Year'] == 1973, 'Productivityvar'].values[0]
gap1973 = productivity1973 - compensation1973
plt.text(1960, 3.75,
         f'1948-1973\n'
         f'$\\mathrm{{Productivity:}}$ {productivity1973*100:.2f}%\n'
         f'$\\mathrm{{Compensation:}}$ {compensation1973*100:.2f}%\n'
         f'$\\mathrm{{GAP:}}$ {gap1973*100:.2f}pp', 
         fontsize=9, fontweight='bold', color='grey', 
         ha='left', va='center')

# Add gap label post 1973
compensation2024 = df.loc[df['Year'] == 2024, 'Compensationvar73'].values[0]
productivity2024 = df.loc[df['Year'] == 2024, 'Productivityvar73'].values[0]
gap2024 = productivity2024 - compensation2024
plt.text(1975, 3.75,
         f'1973-2024\n'
         f'$\\mathrm{{Productivity:}}$ {productivity2024*100:.2f}%\n'
         f'$\\mathrm{{Compensation:}}$ {compensation2024*100:.2f}%\n'
         f'$\\mathrm{{GAP:}}$ {gap2024*100:.2f}pp', 
         fontsize=9, fontweight='bold', color='grey', 
         ha='left', va='center')

# Add comments from 1973
plt.text(1975, 0.25,
         'Expansionary Monetary Policies: $\mathrm{Enhancement\ of\ the\ money\ supply\ and\ reduction\ of\ interest\ rates.}$\n'
         'Globalization: $\mathrm{Market\ liberalization,\ relocation\ of\ production,\ and\ intensified\ international\ competition.}$\n'
         'Economic Structure: $\mathrm{Transition\ from\ a\ manufacturing-based\ model\ to\ one\ emphasizing\ services\ and\ technology.}$',
         fontsize=8, fontweight='bold', color='darkred', 
         ha='left', va='center')

# Configuration grid and labels
plt.xlim(df['Year'].min(), df['Year'].max()+10)
plt.grid(axis='x')
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
plt.text(0.0, 1.05, 'Disconnect between productivity and a typical workers compensation', 
         ha='left', va='bottom', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.0, 1.02, 'United States of America, USA', 
          ha='left', va='bottom', fontsize=12, fontweight='normal', transform=plt.gca().transAxes)
plt.xlabel('Year', fontsize=10, fontweight='bold')
plt.ylabel('Cummulative change since 1948 (%)', fontsize=10, fontweight='bold')
plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='y', labelsize=9) 
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x * 100):,}%'))

# Add Year label 
plt.text(1, 1.07, f'1948–2024',
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Add Data Source
plt.text(0, -0.1, 'Data Source: FRED, Federal Reserve Bank of St. Louis', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Mostrar el gráfico
plt.show()