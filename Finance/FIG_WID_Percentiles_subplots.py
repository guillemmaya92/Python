# Libraries
# ===================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

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

# Filter dataframes
country = ['US', 'FR', 'ES', 'CN', 'WO']
variable = ['sdiincj992', 'shwealj992']
percentile = ['p0p50', 'p90p100']
year = range(1980, 2022)
df = df[(df['country'].isin(country)) & df['variable'].isin(variable) & df['percentile'].isin(percentile) & df['year'].isin(year)]

# Data Manipulation
# ===================================================
# Filtering outliers
df = df[~((df['country'] == 'WO') & 
                   (df['variable'] == 'shwealj992') & 
                   (df['year'] < 2002))]

# Replace values
df['size'] = df['percentile'].replace({'p0p50': '0.5', 'p90p100': '0.1'})
df['percentile'] = df['percentile'].replace({'p0p50': 'Bottom 50', 'p90p100': 'Top 10'})
df['variable'] = df['variable'].replace({'sdiincj992': 'Income', 'shwealj992': 'Wealth'})
df['country'] = df['country'].replace({'CN': 'China', 'FR': 'France', 'US': 'USA', 'ES': 'Spain', 'WO': 'World'})

# Concatenate country and variable
df['variable_percentile'] = df['percentile'].astype(str) + ' (' + df['variable'].astype(str) + ')'

# Selection columns
df = df[['variable_percentile', 'country', 'year', 'size', 'value']]

# Selection columns
df = df[['variable_percentile', 'country', 'year', 'size', 'value']]

print(df)

# Data Visualization
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Color Palette
palette = {
    'China': '#C00000', #Red
    'USA': '#153D64', #Blue
    'France': '#3C7D22', #Green
    'World': '#1C1C1C' #Black
}

# Multiple Line Subplots
g = sns.FacetGrid(df, col='variable_percentile', hue='country', col_wrap=2, 
                  sharey=False, sharex=False, margin_titles=True, despine=False, 
                  palette=palette, height=4, aspect=1.5)
g.map(sns.scatterplot, 'year', 'value', marker='o', s=8, alpha=0.75)
g.map(sns.lineplot, 'year', 'value', linestyle='-', marker='o', markersize=0, linewidth=1, alpha=0.25)

# Title and subtitle
plt.text(-1.12, 2.43, 'Percentiles of Income and Wealth', ha='left', va='top', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(-1.12, 2.36, 'Country Comparison: Evolution of the Bottom 50% and Top 10% (1980-2022)', ha='left', va='top', fontsize=9, color='#262626', transform=plt.gca().transAxes)

# Adjust subplots
plt.subplots_adjust(top=0.9)

# Y Labels (Income / Wealth)
y_labels = ['Income (%)', '', 'Wealth (%)', '']
for ax, label in zip(g.axes.flat, y_labels):
    ax.set_ylabel(label, fontsize=10, fontweight='bold', color='#0D3512')
    
# X Labels (Bottom 50 / Top 10)
if len(g.axes) >= 2:
    g.axes[0].text(0.5, 1.05, 'Bottom 50', ha='center', va='center', transform=g.axes[0].transAxes, fontsize=10, fontweight='bold', color='#0D3512')
    g.axes[1].text(0.5, 1.05, 'Top 10', ha='center', va='center', transform=g.axes[1].transAxes, fontsize=10, fontweight='bold', color='#0D3512')

# X and Y Axis Format
for ax in g.axes.flat:
    ax.set_xlabel('') 
    ax.grid(True, linestyle='-', alpha=0.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    
    # Configure grid
    for spine in ax.spines.values():
        spine.set_edgecolor('#4A4A4A')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.5) 

    # Size X labels
    for label in ax.get_xticklabels():
        label.set_fontsize(8)

    # Size Y labels
    for label in ax.get_yticklabels():
        label.set_fontsize(8)

# Hide titles
g.set_titles('')

# Add Data Source
plt.text(-1.12, -0.15, 'Data Source: World Inequality Database (WID)', 
    transform=plt.gca().transAxes, 
    fontsize=8,
    fontweight='bold',
    color='gray')

# Add Data Source
plt.text(-1.12, -0.21, 'Income: Post-tax national income is the sum of primary incomes over all sectors (private and public), minus taxes.', 
    transform=plt.gca().transAxes, 
    fontsize=7,
    fontstyle='italic',
    color='gray')
plt.text(-1.12, -0.26, 'Wealth: Net personal wealth is the total value of non-financial and financial assets (housing, land, deposits, bonds, equities, etc.) held by households, minus their debts.', 
    transform=plt.gca().transAxes, 
    fontsize=7,
    fontstyle='italic',
    color='gray')

 # Make space for legend
g.add_legend()
g._legend.set_visible(False)

 # Custom Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, lw=1, alpha=0.6, label=region)
    for region, color in palette.items()
]
legend = ax.legend(handles=legend_elements, title='Region', title_fontsize='9', fontsize='8', loc='upper right', bbox_to_anchor=(1.2, 1.25))
plt.setp(legend.get_title(), fontweight='bold')

# Save the figure
plt.savefig('C:/Users/guillem.maya/Desktop/FIG_WID_Percentiles_subplots.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
