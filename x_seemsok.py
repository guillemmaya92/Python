# Libraries
# ===================================================
import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image
from urllib.request import urlopen
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt

# Data Extraction
# ===================================================
# Define CSV path
path = r'C:\Users\guill\Downloads\data\CHECK'

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
dfr = df.copy()
dfp = df.copy()

# Filter dataframes
country = ['ES']
variable = ['sdiincj992', 'shwealj992']
variablev = ['adiincj992', 'ahwealj992']
variablex = ['xlceuxi999']
variablep = ['npopuli999']
percentile = ['p10p100', 'p20p100', 'p30p100', 'p40p100', 'p50p100', 'p60p100', 'p70p100', 'p80p100', 'p90p100']
percentilev = ['p0p100']
year = [2002, 2022]
df = df[(df['country'].isin(country)) & df['variable'].isin(variable) & df['percentile'].isin(percentile) & df['year'].isin(year)]
dfv = dfv[(dfv['country'].isin(country)) & dfv['variable'].isin(variablev) & dfv['percentile'].isin(percentilev) & dfv['year'].isin(year)]
dfr = dfr[(dfr['country'].isin(country)) & dfr['variable'].isin(variablex) & dfr['percentile'].isin(percentilev) & dfr['year'].isin(year)]
dfp = dfp[(dfp['country'].isin(country)) & dfp['variable'].isin(variablep) & dfp['percentile'].isin(percentilev) & dfp['year'].isin(year)]

# Transformation 1
df['value'] = 1 - df['value']
df['percentile'] = df['percentile'].str[1:3].astype(int) / 100
df = df[['country', 'variable', 'year', 'percentile', 'value']]

# Selection columns 2
dfr = dfr[['country', 'year', 'value']]
dfv = pd.merge(dfv, dfr, on=['country', 'year'], how='left')
dfv['value'] = dfv['value_x'] / dfv['value_y'] 
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

# Function to calculate Global GINI
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

df['gini'] = df.groupby(['country', 'variable', 'year'])['value_yr'].transform(lambda x: gini(x))

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
    'gini': 'gini'
    })

df = df[
    ['country', 'variable', 'year', 
     'population_perecent', 'variable_percent', 
     'population_cum_percent', 'variable_cum_percent', 
     'value', 'value_mean', 'value_median', 'gini']
]

# Dataframes Lines
# ===================================================
dfi = df[(df['year'] == 2022) & (df['variable'] == 'income')]
dfip = df[(df['year'] == 2002) & (df['variable'] == 'income')]
dfw = df[(df['year'] == 2022) & (df['variable'] == 'wealth')]
dfwp = df[(df['year'] == 2002) & (df['variable'] == 'wealth')]

data = {
    'population_cum_percent': [0, 1],
    'variable_cum_percent': [0, 1]
}

dfe = pd.DataFrame(data)

# DataFrame Dots
# =====================================================================
# Wealth Dots
dfwavg = dfw.loc[[ (dfw['value'] - dfw['value_mean']).abs().idxmin() ]]
dfwb50 = dfw.loc[[ (dfw['population_cum_percent'] - 0.5).abs().idxmin() ]]
dfwt10 = dfw.loc[[ (dfw['population_cum_percent'] - 0.9).abs().idxmin() ]]

# Income Dots
dfiavg = dfi.loc[[ (dfi['value'] - dfi['value_mean']).abs().idxmin() ]]
dfib50 = dfi.loc[[ (dfi['population_cum_percent'] - 0.5).abs().idxmin() ]]
dfit10 = dfi.loc[[ (dfi['population_cum_percent'] - 0.9).abs().idxmin() ]]

# Data Visualization
# =====================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Create figure and lines
plt.figure(figsize=(10, 10))
plt.plot(dfi['population_cum_percent'], dfi['variable_cum_percent'], label='Income (2022)', color='darkblue')
plt.plot(dfip['population_cum_percent'], dfip['variable_cum_percent'], label='Income (2002)', color='darkblue', linewidth=0.5, linestyle='-')
plt.plot(dfw['population_cum_percent'], dfw['variable_cum_percent'], label='Wealth (2022)', color='darkred')
plt.plot(dfwp['population_cum_percent'], dfwp['variable_cum_percent'], label='Wealth (2002)', color='darkred', linewidth=0.5, linestyle='-')
plt.plot(dfe['population_cum_percent'], dfe['variable_cum_percent'], label='Perfect Distribution', color='darkgrey')

# Title and labels
plt.text(0, 1.05, 'Spain Inequality 2002-2022', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.02, 'Income and Wealth distribution', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income / Wealth (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Wealth Dots 
# ===============
# Add scatter bottom50 wealth
xpop = dfwb50['population_cum_percent'].iloc[0]
ypop = dfwb50['variable_cum_percent'].iloc[0]
vpop = dfwb50['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Population', zorder=5)
plt.text(x=xpop, y=ypop+0.07, 
        s=f'Bottom 50:\n<{vpop: ,.0f}€\n{(ypop) * 100: ,.1f}%', 
        color='darkred', 
        va='center', 
        ha='center',
        fontsize=8)

# Add scatter top10 wealth
xpop = dfwt10['population_cum_percent'].iloc[0]
ypop = dfwt10['variable_cum_percent'].iloc[0]
vpop = dfwt10['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Wealth', zorder=5)
plt.text(x=xpop-0.07, y=ypop, 
         s=f'Top 10:\n>{vpop: ,.0f}€\n{(1-ypop) * 100: ,.0f}%',
         color='darkred', 
         va='center', 
         ha='center', 
         fontsize=8)

# Add scatter average wealth
xpop = dfwavg['population_cum_percent'].iloc[0]
ypop = dfwavg['variable_cum_percent'].iloc[0]
vpop = dfwavg['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='dimgray', label='Mean Wealth', zorder=5, marker='o', facecolor='none')
plt.text(x=xpop-0.07, y=ypop+0.04, 
        s=f'Mean Wealth:\n{vpop: ,.0f}€',  
        color='dimgray', 
        va='center', 
        ha='center', 
        fontsize=8)

# Income Dots 
# ===============
# Add scatter bottom50 income
xpop = dfib50['population_cum_percent'].iloc[0]
ypop = dfib50['variable_cum_percent'].iloc[0]
vpop = dfib50['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkblue', label='Median Population', zorder=5)
plt.text(x=xpop-0.06, y=ypop+0.04, 
        s=f'Bottom 50:\n<{vpop: ,.0f}€\n{(ypop) * 100: ,.1f}%', 
        color='darkblue', 
        va='center', 
        ha='center', 
        fontsize=8)

# Add scatter top10 income
xpop = dfit10['population_cum_percent'].iloc[0]
ypop = dfit10['variable_cum_percent'].iloc[0]
vpop = dfit10['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkblue', label='Median Income', zorder=5)
plt.text(x=xpop-0.08, y=ypop, 
         s=f'Top 10:\n>{vpop: ,.0f}€\n{(1-ypop) * 100: ,.0f}%',
         color='darkblue', 
         va='center', 
         ha='center', 
         fontsize=8)

# Add scatter average income
xpop = dfiavg['population_cum_percent'].iloc[0]
ypop = dfiavg['variable_cum_percent'].iloc[0]
vpop = dfiavg['value'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='dimgray', label='Mean Income', zorder=5, marker='o', facecolor='none')
plt.text(x=xpop-0.06, y=ypop+0.04, 
        s=f'Mean Income:\n{vpop: ,.0f}€',  
        color='dimgray', 
        va='center', 
        ha='center', 
        fontsize=8)

# Gini Legend 
# ===============
# Get Gini Values
Giniw = dfw['gini'].iloc[-1] 
Ginii = dfi['gini'].iloc[-1]
Giniw80 = dfwp['gini'].iloc[-1] 
Ginii80 = dfip['gini'].iloc[-1]

# Add legend
plt.text(0.05, 0.96, f'Gini Wealth (2022): {Giniw:.2f}', color='darkred', fontsize=9, fontweight='bold')
plt.text(0.05, 0.93, f'Gini Wealth (2002): {Giniw80:.2f}', color='darkred', fontsize=9)
plt.text(0.05, 0.90, f'Gini Income (2022): {Ginii:.2f}', color='darkblue', fontsize=9, fontweight='bold')
plt.text(0.05, 0.87, f'Gini Income (2002): {Ginii80:.2f}', color='darkblue', fontsize=9)
plt.text(0.05, 0.84, 'Perfect Distribution: 0', color='darkgrey', fontsize=9, fontweight='bold')

# Configuration
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Add Data Source
plt.text(0, -0.1, 'Data Source: World Inequality Database (WID)', 
    transform=plt.gca().transAxes, 
    fontsize=8,
    fontweight='bold',
    color='gray')

# Add Notes Calculation
plt.text(0, -0.12, 'Notes: The distribution of values, based on Income and Wealth Inequalities, has been smoothed using a monotonic PCHIP interpolator', 
    transform=plt.gca().transAxes,
    fontsize=8,
    fontstyle='italic',
    color='gray')

# Add Notes Income
plt.text(0, -0.14, 'Income: Post-tax national income is the sum of primary incomes over all sectors (private and public), minus taxes.', 
    transform=plt.gca().transAxes,
    fontsize=8,
    fontstyle='italic',
    color='gray')

# Add Notes Wealth
plt.text(0, -0.16, 'Wealth: Net personal wealth is the total value of non-financial and financial assets (housing, land, deposits, bonds, equities, etc.) held by households, minus their debts.', 
    transform=plt.gca().transAxes,
    fontsize=8,
    fontstyle='italic',
    color='gray')

# Add Notes Currency
plt.text(0, -0.18, 'Currency: Official exchange rate of the local currency to EUR.', 
    transform=plt.gca().transAxes,
    fontsize=8,
    fontstyle='italic',
    color='gray')

# Add Flag
url = "https://raw.githubusercontent.com/guillemmaya92/world_flags_round/refs/heads/master/flags/ES.png"
with urlopen(url) as file:
    img_flag = np.array(Image.open(file))
imagebox = OffsetImage(img_flag, zoom=0.05, alpha=0.5)
ab = AnnotationBbox(imagebox, (0.97, 1.04), frameon=False, xycoords='axes fraction')
plt.gca().add_artist(ab)

# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_GINI_Income_Wealth.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
