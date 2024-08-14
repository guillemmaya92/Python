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
parameters = ['NGDPD', 'CG_DEBT_GDP', 'GG_DEBT_GDP', 'HH_LS', 'NFC_LS']

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
df['DEBTper'] = df[['CG_DEBT_GDP', 'GG_DEBT_GDP']].max(axis=1) + df['HH_LS'].fillna(0) + df['NFC_LS'].fillna(0)
df['DEBT'] = df['DEBTper'] * df['NGDPD']
df = df.dropna(subset=['NGDPD', 'DEBTper'], how='any')

# Merge queries
df = df.merge(df_countries, how='left', left_on='ISO3', right_on='ISO3')
df = df[['ISO3', 'Year', 'NGDPD', 'DEBTper', 'DEBT', 'Region']]
df = df[df['Region'].notna()]

# Filter nulls and order
df = df.sort_values(by=['Year', 'DEBTper'])

# Calculate 'left accrual widths'
df['NGDPDcum'] = df.groupby('Year')['NGDPD'].cumsum()
df['Left'] = df.groupby('Year')['NGDPD'].cumsum() - df['NGDPD']

# Calculate global
dfg = df.copy()
dfg = dfg.groupby('Year').agg({'NGDPD': 'sum', 'DEBT': 'sum'}).reset_index()
dfg['DEBTG_per'] = dfg['DEBT'] / dfg['NGDPD']
dfg['DEBTG_var'] = dfg['DEBTG_per'] - dfg['DEBTG_per'].iloc[0]
dfg = dfg[['Year', 'DEBTG_per', 'DEBTG_var']]

# Merge queries
df = df.merge(dfg, how='left', on='Year')

print(df)

# Data Visualization
# =====================================================================
# Seaborn figure style
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(16, 9))

# Custom palette
custom = {
    'Asia': '#fbecc3',
    'Europe': '#bcd2be', 
    'Oceania': '#bcccdb', 
    'Americas': '#fbbcbc',
    'Africa': '#ffc394'
}

# Create legend lines
legend_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in custom.values()]
legend_labels = list(custom.keys())

# Function to refresh animation
def update(year):
    plt.clf()
    subset = df[df['Year'] == year]
    
    # Create a Matplotlib plot
    bars = plt.bar(subset['Left'], subset['DEBTper'], width=subset['NGDPD'], 
           color=subset['Region'].map(custom), alpha=1, align='edge', edgecolor='grey', linewidth=0.1)
    
    # Configuration grid and labels
    plt.xlim(0, subset['NGDPDcum'].max())
    plt.ylim(0, 500)
    plt.grid(axis='x')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    plt.title(f'Total Debt Distribution by Country', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cumulative GDP (M)', fontsize=10, fontweight='bold')
    plt.ylabel('Global Debt Percent of GDP (%)', fontsize=10, fontweight='bold')
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9) 
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        
    # Add Labels to relevant countries
    for bar, value, country in zip(bars, subset['NGDPD'], subset['ISO3']):
        if country in ['CHN', 'IND', 'USA', 'IDN', 'PAK', 'NGA', 'BRA', 'BGD', 'RUS', 'MEX', 'JPN', 'VNM', 'DEU', 'GBR', 'FRA', 'ITA', 'CAN', 'AUS']:
            plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{country}\n{''}', ha='center', va='bottom', fontsize=7, color='grey')

    # Add Year label 
    plt.text(0.95, 1.06, f'{year}',
             transform=plt.gca().transAxes,
             fontsize=22, ha='right', va='top',
             fontweight='bold', color='#D3D3D3')
    
    # Add Data Source
    plt.text(0, -0.1, 'Data Source: IMF World Economic Outlook Database, 2024', 
            transform=plt.gca().transAxes, 
            fontsize=8, 
            color='gray')
    
    # Add Global Debt
    debtg_per = subset['DEBTG_per'].max()
    debtg_var = subset['DEBTG_var'].max()
    
    plt.axhline(
        y=debtg_per,
        color='#333333', 
        linestyle='--', 
        linewidth=0.5,
        label=f'Median: {debtg_per:,.0f}')
    
    plt.text(
        x=subset['Left'].max() * 0.02,
        y=debtg_per + (500 * 0.04),
        s=f'Global Debt: {debtg_per:,.2f}%',
        color='#333333',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        weight='bold')

    plt.text(
        x=subset['Left'].max() * 0.02,
        y=debtg_per + (500 * 0.02),
        s=f'Cummulative Var.: {debtg_var:,.2f}pp',
        color='darkgreen',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=9,
        weight='bold')

    # Add Legend
    legend = plt.legend(legend_lines, legend_labels, title='Region', loc='upper left', title_fontsize='10', fontsize='9', bbox_to_anchor=(0, 1))
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Add label "Lowest" and "Highest"
    plt.text(0, -0.065, 'Lowest',
             transform=ax.transAxes,
             fontsize=11, fontweight='bold', color='#8E99C8', ha='left', va='center')
    plt.text(0.95, -0.065, 'Highest',
             transform=ax.transAxes,
             fontsize=11, fontweight='bold', color='#D08686', va='center')

# Configurate animation
years = sorted(df['Year'].unique())
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, interval=250, blit=False)

# Save the animation :)
ani.save('C:/Users/guill/Downloads/FIG_DEBT_GDP_Bars.webp', writer='imagemagick', fps=3)

# Print it!
plt.show()