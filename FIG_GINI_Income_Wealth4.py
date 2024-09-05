import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# Data Extraction 
# =====================================================
# Open csv file as dataframe
file_path = r"C:\Users\guillem.maya\Downloads\data\WID_data_ES.csv"
df = pd.read_csv(file_path, delimiter=';')

# Filter dataframe
variable = ['shwealj992', 'scaincj992', 'sdiincj992']
percentile = ['p10p100', 'p20p100', 'p30p100', 'p40p100', 'p50p100', 'p60p100', 'p70p100', 'p80p100', 'p90p100']
year = [1985, 2022]
df = df[df['variable'].isin(variable) & df['percentile'].isin(percentile) & df['year'].isin(year)]

# Transformation
df['value'] = 1 - df['value']
df['percentile'] = df['percentile'].str[1:3].astype(int) / 100
df = df[['variable', 'year', 'percentile', 'value']]

# Add values 0 and 1
dfx = pd.DataFrame(
    [(v, y, 0, 0) for v in variable for y in year] + 
    [(v, y, 1, 1) for v in variable for y in year],
    columns=['variable', 'year', 'percentile', 'value']
)
df = pd.concat([df, dfx], ignore_index=True)
df = df.sort_values(by=['variable', 'year', 'percentile']).reset_index(drop=True)

# Lambda for create arrays
grouped = df.groupby(['variable', 'year'])
grouped_data = df.groupby(['variable', 'year'])[['percentile', 'value']].apply(lambda x: x.to_numpy())

# Get minimum and maximum years
min_year = df['year'].min()
max_year = df['year'].max()

# Create arrays
percentile = grouped_data.get(('shwealj992', max_year))[:, 0]
wealthp = grouped_data.get(('shwealj992', min_year))[:, 1]
wealthf = grouped_data.get(('shwealj992', max_year))[:, 1]
incomep = grouped_data.get(('scaincj992', min_year))[:, 1]
incomef = grouped_data.get(('scaincj992', max_year))[:, 1]
incomep2 = grouped_data.get(('sdiincj992', min_year))[:, 1]
incomef2 = grouped_data.get(('sdiincj992', max_year))[:, 1]

# Data Extraction 
# =====================================================
# Open csv file as dataframe
dfv = pd.read_csv(file_path, delimiter=';')

# Filter dataframe
variablev = ['adiincj992', 'ahwealj992']
percentilev = ['p0p100']
yearv = [2022]
dfv = dfv[dfv['variable'].isin(variablev) & dfv['percentile'].isin(percentilev) & dfv['year'].isin(yearv)]

# Transformation
dfv = dfv[['variable', 'year', 'percentile', 'value']]
wealthvalue = dfv.iloc[0]['value']
incomevalue = dfv.iloc[1]['value']

# Gini Function
# =====================================================
# Function to calculate Global GINI
def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    gini_index = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini_index

# Income Inequality (2022)
# =====================================================
# Datos de entrada
x = percentile
y = incomef2

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 1000)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfi = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Add Income and adjust
dfi['Income'] = (dfi['y_suave'] - dfi['y_suave'].shift(1)).fillna(0) * dfi.shape[0] * incomevalue

# Add Mean and median
dfi['Mean'] = dfi['Income'].mean()
dfi['Median'] = dfi['Income'].median()
dfi['Gini'] = gini(dfi['Income'])

# Income Inequality (1985)
# =====================================================
# Datos de entrada
x = percentile
y = incomep2

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 1000)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfi80 = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Add Income and adjust
dfi80['Income'] = (dfi80['y_suave'] - dfi80['y_suave'].shift(1)).fillna(0) * dfi80.shape[0]

# Add Mean and median
dfi80['Mean'] = dfi80['Income'].mean()
dfi80['Median'] = dfi80['Income'].median()
dfi80['Gini'] = gini(dfi80['Income'])

# Wealth Inequality (2022)
# =====================================================
# Datos de entrada
x = percentile
y = wealthf

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 1000)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfw = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Add Wealth and adjust
dfw['Wealth'] = (dfw['y_suave'] - dfw['y_suave'].shift(1)).fillna(0) * dfw.shape[0] * wealthvalue

# Add Mean and median
dfw['Mean'] = dfw['Wealth'].mean() 
dfw['Median'] = dfw['Wealth'].median()
dfw['Gini'] = gini(dfw['Wealth'])

# Wealth Inequality (1985)
# =====================================================
# Datos de entrada
x = percentile
y = wealthp

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 1000)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfw80 = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Add Wealth and adjust
dfw80['Wealth'] = (dfw80['y_suave'] - dfw80['y_suave'].shift(1)).fillna(0) * dfw80.shape[0] * 209825

# Add Mean and median
dfw80['Mean'] = dfw80['Wealth'].mean() 
dfw80['Median'] = dfw80['Wealth'].median()
dfw80['Gini'] = gini(dfw80['Wealth'])

# DataFrame Equality
# =====================================================================
data = {
    'POP_Cum': [0, 1],
    'GDP_Cum': [0, 1]
}

dfe = pd.DataFrame(data)

# DataFrame Row Mean and Median
# =====================================================================
# Wealth Dots
dfmean = dfw.loc[[ (dfw['Wealth'] - dfw['Mean']).abs().idxmin() ]]
dfmedianpop = dfw.loc[[ (dfw['x_suave'] - 0.5).abs().idxmin() ]]
dfmedianinc = dfw.loc[[ (dfw['y_suave'] - 0.5).abs().idxmin() ]]

# Income Dots
dfmean2 = dfi.loc[[ (dfi['Income'] - dfi['Mean']).abs().idxmin() ]]
dfmedianpop2 = dfi.loc[[ (dfi['x_suave'] - 0.5).abs().idxmin() ]]
dfmedianinc2 = dfi.loc[[ (dfi['y_suave'] - 0.5).abs().idxmin() ]]

# Data Visualization
# =====================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Create figure and lines
plt.figure(figsize=(10, 10))
plt.plot(dfi['x_suave'], dfi['y_suave'], label='Income (2022)', color='darkblue')
plt.plot(dfi80['x_suave'], dfi80['y_suave'], label='Income (1985)', color='darkblue', linewidth=0.5, linestyle='-')
plt.plot(dfw['x_suave'], dfw['y_suave'], label='Wealth (2022)', color='darkred')
plt.plot(dfw80['x_suave'], dfw80['y_suave'], label='Wealth (1985)', color='darkred', linewidth=0.5, linestyle='-')
plt.plot(dfe['POP_Cum'], dfe['GDP_Cum'], label='Perfect Distribution', color='darkgrey')

# Wealth Dots 
# ===============
# Add scatter median population
xpop = dfmedianpop['x_suave'].iloc[0]
ypop = dfmedianpop['y_suave'].iloc[0]
vpop = dfmedianpop['Wealth'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Population', zorder=5)
plt.text(x=xpop, y=ypop+0.05, 
        s=f'Median Population:\n{vpop: ,.0f}€\nB{(xpop) * 100: ,.0f}%-{(ypop) * 100: ,.1f}%', 
        color='darkred', 
        va='center', 
        ha='center', 
        fontsize=8)

# Add scatter median income
xpop = dfmedianinc['x_suave'].iloc[0]
ypop = dfmedianinc['y_suave'].iloc[0]
vpop = dfmedianinc['Wealth'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkred', label='Median Wealth', zorder=5)
plt.text(x=xpop-0.055, y=ypop, 
         s=f'Median Wealth:\n{vpop: ,.0f}€\nT{(1-xpop) * 100: ,.1f}%-{(ypop) * 100: ,.0f}%',
         color='darkred', 
         va='center', 
         ha='center', 
         fontsize=8)

# Add scatter mean income
xpop = dfmean['x_suave'].iloc[0]
ypop = dfmean['y_suave'].iloc[0]
vpop = dfmean['Mean'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='dimgray', label='Mean Wealth', zorder=5, marker='o', facecolor='none')
plt.text(x=xpop-0.05, y=ypop+0.03, 
        s=f'Mean Wealth:\n{vpop: ,.0f}€\nB{(xpop) * 100: ,.0f}%-{(ypop) * 100: ,.0f}%',  
        color='dimgray', 
        va='center', 
        ha='center', 
        fontsize=8)

# Income Dots 
# ===============
# Add scatter median population
xpop = dfmedianpop2['x_suave'].iloc[0]
ypop = dfmedianpop2['y_suave'].iloc[0]
vpop = dfmedianpop2['Income'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkblue', label='Median Population', zorder=5)
plt.text(x=xpop-0.06, y=ypop+0.06, 
        s=f'Median Population:\n{vpop: ,.0f}€\nB{(xpop) * 100: ,.0f}%-{(ypop) * 100: ,.1f}%', 
        color='darkblue', 
        va='center', 
        ha='center', 
        fontsize=8)

# Add scatter median income
xpop = dfmedianinc2['x_suave'].iloc[0]
ypop = dfmedianinc2['y_suave'].iloc[0]
vpop = dfmedianinc2['Income'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='darkblue', label='Median Income', zorder=5)
plt.text(x=xpop-0.05, y=ypop+0.05, 
         s=f'Median Income:\n{vpop: ,.0f}€\nT{(1-xpop) * 100: ,.1f}%-{(ypop) * 100: ,.0f}%',
         color='darkblue', 
         va='center', 
         ha='center', 
         fontsize=8)

# Add scatter mean income
xpop = dfmean2['x_suave'].iloc[0]
ypop = dfmean2['y_suave'].iloc[0]
vpop = dfmean2['Mean'].iloc[0]
plt.scatter(x=xpop, y=ypop, color='dimgray', label='Mean Income', zorder=5, marker='o', facecolor='none')
plt.text(x=xpop-0.06, y=ypop+0.04, 
        s=f'Mean Income:\n{vpop: ,.0f}€\nB{(xpop) * 100: ,.0f}%-{(ypop) * 100: ,.0f}%',  
        color='dimgray', 
        va='center', 
        ha='center', 
        fontsize=8)

# Gini Legend 
# ===============
# Get Gini Values
Giniw = dfw['Gini'].iloc[-1] 
Ginii = dfi['Gini'].iloc[-1]
Giniw80 = dfw80['Gini'].iloc[-1] 
Ginii80 = dfi80['Gini'].iloc[-1]

# Add legend
plt.text(0.05, 0.96, f'Gini Wealth (2022): {Giniw:.2f}', color='darkred', fontsize=9, fontweight='bold')
plt.text(0.05, 0.93, f'Gini Wealth (1985): {Giniw80:.2f}', color='darkred', fontsize=9)
plt.text(0.05, 0.90, f'Gini Income (2022): {Ginii:.2f}', color='darkblue', fontsize=9, fontweight='bold')
plt.text(0.05, 0.87, f'Gini Income (1985): {Ginii80:.2f}', color='darkblue', fontsize=9)
plt.text(0.05, 0.84, 'Perfect Distribution: 0', color='darkgrey', fontsize=9, fontweight='bold')

# Title and labels
plt.suptitle('   Spain Inequality 1985-2022', fontsize=16, fontweight='bold', y=0.95)
plt.title('Income and Wealth distribution', fontsize=12, fontweight='bold', color='darkgrey', pad=20)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income / Wealth (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Configuration
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Add Year label 
plt.text(1, 1.06, f'2022',
    transform=plt.gca().transAxes,
    fontsize=16, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Add Data Source
plt.text(0, -0.1, 'Data Source: World Inequality Database (WID)', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Add Notes
plt.text(0, -0.12, 'Notes: The distribution of values, based on Income and Wealth Inequalities, has been smoothed using a monotonic PCHIP interpolator', 
    transform=plt.gca().transAxes,
    fontsize=8, 
    color='gray')

# Save the figure
plt.savefig('C:/Users/guillem.maya/Desktop/FIG_GINI_Income_Wealth.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()