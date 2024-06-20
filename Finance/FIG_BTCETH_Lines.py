# Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=BTC&limit=365&toTs={to_ts}"
    response = requests.get(url)
    data = response.json().get("Data", {}).get("Data", [])
    df = pd.DataFrame([
        {
            "symbol": "BTCETH",
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
           'date': ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'] 
           }

halving = pd.DataFrame(halving)
halving['date'] = pd.to_datetime(halving['date'])

# DataSet 1 - BTC Price
# ==============================================================================
# Definir y ordenar dataset
btc = btc.drop_duplicates()
btc['date'] = pd.to_datetime(btc['date'])
btc['year_month'] = btc['date'].dt.strftime('%Y-%m')
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
btc = btc[btc['halving'] >= 2]
btc['daystotal'] = btc.groupby('symbol').cumcount() + 1

# Graph 1 - SEABORN
# ==============================================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Colors Background
regions = [
    (0, 500, '#6B8E23'), # Green
    (500, 1000, '#FF4500'), # Red
    (1000, 1500, '#FFA500') # Orange
    ]  

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
sns.lineplot(x='days', y='close', hue='halving', data=btc, markers=True, palette=lines, linewidth=1)

# Add region colors in the background
for start, end, color in regions:
    plt.axvspan(start, end, color=color, alpha=0.05)

# Title and axis
plt.title('ETH/BTC: Analysis Since Each Halving', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Days',  fontsize=10, fontweight='bold')
plt.ylabel('ETH/BTC', fontsize=10, fontweight='bold')
plt.xlim(0, 1500)
plt.xticks(range(0, 1501, 125), fontsize=9)
plt.tick_params(axis='both', labelsize=8)
plt.yticks(fontsize=9)

# Custom legend
legend = plt.legend(title="Halving", loc='lower right', fontsize=8, title_fontsize='10')
new_title = 'Dates:'
legend.set_title(new_title)
new_labels = ['1st Halving: 2012-11-28 to 2016-07-09', '2nd Halving: 2016-07-09 to 2020-05-11', '3rd Halving: 2020-05-11 to 2024-04-20', '4th Halving: 2024-04-20 to present'] # Adjust the number of labels according to your data
for text, new_label in zip(legend.texts, new_labels):
    text.set_text(new_label)

# Max First 750 days
btc1 = btc[(btc['days'] >= 0) & (btc['days'] <= 750)]
for halving, group in btc1.groupby('halving'):
    max_value = group['close'].max()
    max_row = group[group['close'] == max_value].iloc[0]
    plt.plot(max_row['days'], max_row['close'] +0.0025, marker='*', color='darkgoldenrod', markersize=5)
    plt.text(max_row['days'], max_row['close'] +0.005, f'{max_row["close"]:,.4f} ({max_row["increase"]:,.0f})%', fontsize=7, ha='center', color='darkgoldenrod')


# Min After 750 days 
btc3 = btc[(btc['days'] >= 750) & (btc['days'] <= 1500)]
for halving, group in btc3.groupby('halving'):
    min_value = group['close'].min()
    min_row = group[group['close'] == min_value].iloc[0]
    plt.plot(min_row['days'], min_row['close'] -0.0015, marker='v', color='darkred', markersize=5)
    plt.text(min_row['days'], min_row['close'] -0.005, f'{min_row["close"]:,.4f} ({min_row["increase"]:,.0f})%', fontsize=7, ha='center', color='darkred')

# Custom Last Dots
max_vals = btc.groupby('halving').agg({'close': 'last', 'days': 'max'}).reset_index()
for index, row in max_vals.iterrows():
    plt.plot(row['days'], row['close'], 'ro', markersize=2)

# Custom Line labels
for halving, group in btc.groupby('halving'):
    last_point = group.iloc[-1]
    x = last_point['days']
    y = last_point['close']
    max_days = group['days'].max()
    plt.text(x + 8, y, f'Halving {halving}\n{max_days} days', color=lines[halving], fontsize=8, ha='left', va='center')
    
# Add Year Label
current_year_month = btc['year_month'].max() 
plt.text(1, 1.05, f'{current_year_month}',
    transform=plt.gca().transAxes,
    fontsize=22, ha='right', va='top',
    fontweight='bold', color='#D3D3D3')

# Add Data Source
plt.text(0, -0.065, 'Data Source: CryptoCompare', 
         transform=plt.gca().transAxes, 
         fontsize=8, 
         color='gray')

# Adjust layout
plt.tight_layout()

# Print it!
plt.show()