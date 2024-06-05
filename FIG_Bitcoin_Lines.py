# Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as p9
import requests

# Get API Data
# ==============================================================================
# Create a df with final year dates
dp = pd.DataFrame({'date': pd.date_range(start='2010-12-31', end='2024-12-31', freq='Y')})
dp['to_ts'] = dp['date'].apply(lambda x: int(pd.to_datetime(x).timestamp()))

# Create an empty list
dataframes = []

# Iterate API with each date
for to_ts in dp['to_ts']:
    # Build an URL with parameters and transform data
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=365&toTs={to_ts}"
    response = requests.get(url)
    data = response.json().get("Data", {}).get("Data", [])
    df = pd.DataFrame([
        {
            "symbol": "BTCUSD",
            "date": pd.to_datetime(entry["time"], unit="s").date(),
            "open": entry["open"],
            "close": entry["close"],
            "low": entry["low"],
            "high": entry["high"],
            "volume": entry["volumeto"]
        }
        for entry in data
    ])
    dataframes.append(df)
# Combine all df into one
btc = pd.concat(dataframes, ignore_index=True)

# DataSet 0 - Halving
#================================================================================
halving = {'halving': [0 , 1, 2, 3, 4],
               'date': ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'] }

halving = pd.DataFrame(halving)
halving['date'] = pd.to_datetime(halving['date'])

# DataSet 1 - BTC Price
# ==============================================================================
# Definir y ordenar dataset
btc = btc.drop_duplicates()
btc['date'] = pd.to_datetime(btc['date'])
btc = btc.set_index('date')
btc = btc.asfreq('D').ffill()
btc = btc.reset_index()
btc.sort_values(by=['date'], inplace=True)
btc = pd.merge(btc, halving, on='date', how='left')
btc['halving'].fillna(method='ffill', inplace=True)
btc['halving'].fillna(0, inplace=True)
btc['halving'] = btc['halving'].astype(int)
btc['first_close'] = btc.groupby('halving')['close'].transform('first')
btc['increase'] = (btc['close'] - btc['first_close']) / btc['first_close'] * 100
btc['days'] = btc.groupby('halving').cumcount() + 1
btc['closelog'] = np.log10(btc['close'])
btc = btc[btc['halving'] >= 1]

# Graph 1 - PLOTNINE
# ==============================================================================
# Plot a graph
# Calculate maximum closelog and days for each halving
max_vals = btc.groupby('halving').agg({'closelog': 'last', 'days': 'max'}).reset_index()

# Dictionary to hold halving labels
halving_labels = {
    0: 'Introduction',
    1: '1st Halving',
    2: '2nd Halving',
    3: '3rd Halving',
    4: '4th Halving'
}

plot = (
    p9.ggplot(btc, p9.aes(x='days', y='closelog', color='factor(halving)'))
    + p9.geom_rect(
        p9.aes(xmin=0, xmax=500, ymin=-float('inf'), ymax=float('inf')),
        fill='#F5FFFA',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_rect(
        p9.aes(xmin=500, xmax=1000, ymin=-float('inf'), ymax=float('inf')),
        fill='#FFF2F2',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_rect(
        p9.aes(xmin=1000, xmax=1500, ymin=-float('inf'), ymax=float('inf')),
        fill='#FEFDF9',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_line(size=0.5)
    + p9.labs(
        title='BTC Log Price - From Each Halving',
        x='Days',
        y='Log Price',
        color='Halving'
    )
    + p9.xlim(0, 1500)
    + p9.theme_minimal()
    + p9.scale_color_manual(
        name='Halving',
        values = {
            0: '#E0E0E0',  # Very Light Grey
            1: '#C0C0C0',  # Light Grey
            2: '#808080',  # Medium Grey
            3: '#404040',  # Dark Grey
            4: '#8B0000'   # Red
        },
        labels = {
            0: 'Int: 2009-01-03 to 2012-11-28', # Introdution
            1: '1st: 2012-11-28 to 2016-07-09', # 1st Halving
            2: '2nd: 2016-07-09 to 2020-05-11', # 2nd Halving
            3: '3rd: 2020-05-11 to 2024-04-20', # 3rd Halving
            4: '4th: 2024-04-20 to present' # 4th Halving
        }
    )
    + [p9.annotate('text', x=row['days']+60, y=row['closelog'], label=halving_labels[row['halving']], ha='center', va='top', size=8) for idx, row in max_vals.iterrows()]
    + p9.theme(
        figure_size=(16, 9),
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        text=p9.element_text(family="Open Sans"),
        plot_title=p9.element_text(size=14, weight='bold'),
        panel_background=p9.element_rect(fill="white", color="white")
    )
)

# Print it!
print(plot)

# Graph 2 - SEABORN
# ==============================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Colors Background
regions = [
    (0, 500, '#6B8E23'),  # Green
    (500, 1000, '#FF4500'),  # Red
    (1000, 1500, '#FFA500')
    ]  # Orange

# Colors Palette Lines
lines = {
    0: '#E0E0E0',  # Very Light Grey
    1: '#C0C0C0',  # Light Grey
    2: '#808080',  # Medium Grey
    3: '#404040',  # Dark Grey
    4: '#8B0000'   # Red
}

# Seaborn to plot a graph
sns.set(style="whitegrid", rc={"grid.color": "0.95", "axes.grid.axis": "y"})
plt.figure(figsize=(16, 9))
sns.lineplot(x='days', y='closelog', hue='halving', data=btc, markers=True, palette=lines, linewidth=1)

# Add region colors in the background
for start, end, color in regions:
    plt.axvspan(start, end, color=color, alpha=0.05)

# Title and axis
plt.title('BTC Log Price - From Each Halving', fontsize=16, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Log Price')
plt.xlim(0, 1500)
plt.ylim(0.75, 5.25)

# Custom legend
legend = plt.legend(title="Halving", loc='lower right', fontsize=8, title_fontsize='10')
new_title = 'Dates:'
legend.set_title(new_title)
new_labels = ['1st Halving: 2012-11-28 to 2016-07-09', '2nd Halving: 2016-07-09 to 2020-05-11', '3rd Halving: 2020-05-11 to 2024-04-20', '4th Halving: 2024-04-20 to present'] # Adjust the number of labels according to your data
for text, new_label in zip(legend.texts, new_labels):
    text.set_text(new_label)

# Custom line labels
for halving, group in btc.groupby('halving'):
    last_point = group.iloc[-1]
    x = last_point['days']
    y = last_point['closelog']
    plt.text(x + 20, y, f'Halving {halving}', color=lines[halving], fontsize=8, ha='left', va='center')

# Custom Maximum Dots
for halving, group in btc.groupby('halving'):
    max_value = group['closelog'].max()
    max_row = group[group['closelog'] == max_value].iloc[0]
    plt.plot(max_row['days'], max_row['closelog']+0.05, marker='*', color='darkgoldenrod', markersize=5)

# Custom Last Dots
max_vals = btc.groupby('halving').agg({'closelog': 'last', 'days': 'max'}).reset_index()
for index, row in max_vals.iterrows():
    plt.plot(row['days'], row['closelog'], 'ro', markersize=2)

# Adjust layout
plt.tight_layout()

# Print it!
plt.show()
