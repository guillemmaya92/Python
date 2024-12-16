# Libraries
# ===================================================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

# Data Extraction
# ===================================================
# Define CSV path
path = r'C:\Users\guillem.maya\Downloads\data\Y'

# List to save dataframe
list = []

# Iterate over each file
for archivo in os.listdir(path):
    if archivo.startswith("WID_data_") and archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, archivo), delimiter=';')
        list.append(df)

# Combine all dataframe
df = pd.concat(list, ignore_index=True)

# Data Manipulation
# ===================================================
# Filter dataframe
variable = ['shwealj992', 'ahwealj992']
year = [2022]
df = df[df['variable'].isin(variable) & df['year'].isin(year)]

# Clean dataframe
df = df[~df['percentile'].str.contains(r'\.', na=False)]
df['dif'] = df['percentile'].str.extract(r'p(\d+)p(\d+)').astype(int).apply(lambda x: x[1] - x[0], axis=1)
df = df[df['dif'] == 1]
df['percentile'] = df['percentile'].str.extract(r'p\d+p(\d+)').astype(int)
df = df.sort_values(by='percentile')

# Pivot dataframe
df['variable'] = df['variable'].replace({'shwealj992': 'percentage', 'ahwealj992': 'value'})
df = df[['variable', 'year', 'country', 'percentile', 'value']]
df = df.pivot_table(index=['year', 'country', 'percentile'], columns='variable', values='value')
df = df.reset_index()

# Grouping by 10
df['percentile2'] = pd.cut(
    df['percentile'], 
    bins=range(1, 111, 10), 
    right=False, 
    labels=[i + 9 for i in range(1, 101, 10)]
).astype(int)

# Define palette
color_palette = {
    10: "#050407",
    20: "#07111e",
    30: "#15334b",
    40: "#2b5778",
    50: "#417da1",
    60: "#5593bb",
    70: "#5a7aa3",
    80: "#6d5e86",
    90: "#a2425c",
    100: "#D21E00"
}

# Map palette color
df['color'] = df['percentile2'].map(color_palette)

# Percentiles dataframe
df2 = df.copy()
df2 = df.groupby(['percentile2', 'color'], as_index=False)['value'].sum()
df2['valueper'] = df2['value'] / (df2['value']).sum()
df2['count'] = 10

print(df)

# Data Visualization
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Create the figure and suplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [10, 1]})

# First Plot
# ==================
# Plot Bars
bars = ax1.bar(df['percentile'], df['value'], color=df['color'], edgecolor='darkgrey', linewidth=0.5)

# Title and labels
ax1.text(0, 1.1, 'Global Wealth Distribution', fontsize=13, fontweight='bold', ha='left', transform=ax1.transAxes)
ax1.text(0, 1.06, 'By percentiles of population', fontsize=9, color='#262626', ha='left', transform=ax1.transAxes)
ax1.set_xlabel('% Population', fontsize=10, weight='bold')
ax1.set_ylabel('Wealth (€)', fontsize=10, weight='bold')

# Configuration
ax1.grid(axis='x', linestyle='-', alpha=0.5)
ax1.set_xlim(0, 101)
ax1.set_ylim(0, 350000)
ax1.set_xticks(np.arange(0, 101, step=10))
ax1.set_yticks(np.arange(0, 350001, step=50000))
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Formatting x and y axis
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x / 1e3)))

# Lines and area to separate outliers
ax1.axhline(y=330000, color='black', linestyle='--', linewidth=0.5, zorder=3)
ax1.axhline(y=320000, color='black', linestyle='--', linewidth=0.5, zorder=3)
ax1.add_patch(patches.Rectangle((0, 320000), 105, 10000, linewidth=0, edgecolor='none', facecolor='white', zorder=2))

# Y Axis modify the outlier value
labels = [item.get_text() for item in ax1.get_yticklabels()]
labels[-1] = '1000K'
ax1.set_yticklabels(labels)

# Show labels each 10 percentile
for i, (bar, value) in enumerate(zip(bars, df['value'])):
    if i % 10 == 0 and i != 0:
        ax1.text(bar.get_x() + bar.get_width() / 2, 
                 abs(bar.get_height()) * 1.25 + 5000,
                 f'{value:,.0f}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=8,
                 color='black', 
                 rotation=90)

# Second Plot
# ==================
# Plot Bars
ax2.barh([0] * len(df2), df2['count'], left=df2['percentile2'] - df2['count'], color=df2['color'])

# Configuration
ax2.grid(axis='x', linestyle='-', color='white', alpha=1, linewidth=0.5)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
x_ticks = np.linspace(df2['percentile2'].min(), df2['percentile2'].max(), 10)
ax2.set_xticks(x_ticks)
ax2.set_xlim(0, 101)

# Add label values
for i, row in df2.iterrows():
    plt.text(row['percentile2'] - row['count'] + row['count'] / 2, 0, 
             f'{row["valueper"] * 100:.2f}%', ha='center', va='center', color='white', fontweight='bold')
    

 # Add Year label
formatted_date = 2022 
ax1.text(1, 1.08, f'{formatted_date}',
    transform=ax1.transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Add Data Source
ax2.text(0, -0.5, 'Data Source: World Inequality Database (WID)', 
    transform=ax2.transAxes, 
    fontsize=8, 
    color='gray')

# Adjust layout
plt.tight_layout()

# Save it...
plt.savefig("C:/Users/guillem.maya/Downloads/FIG_WID_Global_Wealth_Distribution.png", dpi=300, bbox_inches='tight') 

# Plot it!
plt.show()