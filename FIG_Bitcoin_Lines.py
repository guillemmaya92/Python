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

# Graph 1
# ==============================================================================
# Plot a graph
plot = (
    p9.ggplot(btc, p9.aes(x='days', y='closelog', color='factor(halving)'))
    + p9.geom_rect(
        p9.aes(xmin=0, xmax=500, ymin=-float('inf'), ymax=float('inf')),
        fill='#E0FFE0',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_rect(
        p9.aes(xmin=500, xmax=1000, ymin=-float('inf'), ymax=float('inf')),
        fill='#FFE0E0',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_rect(
        p9.aes(xmin=1000, xmax=1500, ymin=-float('inf'), ymax=float('inf')),
        fill='#FFF0E0',
        alpha=0.1,
        inherit_aes=False
    )
    + p9.geom_line(size=1)
    + p9.labs(
        title='BTC Price - From Each Halving',
        x='Days',
        y='Log Price',
        color='Halving'
    )
    + p9.theme_minimal()
    + p9.theme(
        figure_size=(10, 6),
        axis_text_x=p9.element_text(rotation=45, hjust=1)
    )
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
            0: 'Int: 2009-01-03 to 2012-11-28',
            1: '1st: 2012-11-28 to 2016-07-09',
            2: '2nd: 2016-07-09 to 2020-05-11',
            3: '3rd: 2020-05-11 to 2024-04-20',
            4: '4th: 2024-04-20 to present'
        }
    )
)

# Print it!
print(plot)
