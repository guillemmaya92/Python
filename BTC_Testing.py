import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objects as go
import plotly.io as pio

# Info Halvings
# ==============================================================================
btc_halving = {'halving'              : [0, 1 , 2, 3, 4, 5],
               'date'                 : ['2009-01-03', '2012-11-28', 
                                         '2016-07-09', '2020-05-11', 
                                         '2024-04-20', np.nan],
               'reward'               : [50, 25, 12.5, 6.25, 3.125, 1.5625],
               'halving_block_number' : [0, 210000, 420000 ,630000, 840000, 1050000]
              }

# Get block height
# ==============================================================================
def get_block_height():
    response = requests.get("https://blockchain.info/q/getblockcount")
    block_height = int(response.text)
    return block_height
block_height = get_block_height()

# Average blocks per minute
# ==============================================================================
def get_block_interval_minute():
        response = requests.get("https://blockchain.info/q/interval")
        interval_seconds = float(response.text)
        interval_minutes = interval_seconds / 60
        return interval_minutes
block_minute = get_block_interval_minute()

# Calculate next halving
# ==============================================================================
remaining_blocks = max(btc_halving['halving_block_number']) - block_height
blocks_per_day = 1440 / block_minute

remaining_days = remaining_blocks / blocks_per_day

next_halving = pd.to_datetime(datetime.date.today(), format='%Y-%m-%d') + datetime.timedelta(days=remaining_days)
next_halving = next_halving.replace(microsecond=0, second=0, minute=0, hour=0)
next_halving = next_halving.strftime('%Y-%m-%d')

btc_halving['date'][-1] = next_halving

print(f'The next halving will occur approximately on: {next_halving}')

# Get BTC values
# ==============================================================================
import get_df_crypto
data = get_df_crypto.df
data = data.set_index('date')
data = data.sort_index()

# Incluir recompensas y cuenta regresiva para próximo halving en el dataset
# ==============================================================================
data['reward'] = np.nan
data['countdown_halving'] = np.nan

for i in range(len(btc_halving['halving'])-1):
     
    # Fecha inicial y final de cada halving
    if btc_halving['date'][i] < data.index.min().strftime('%Y-%m-%d'):
        start_date = data.index.min().strftime('%Y-%m-%d')
    else:
        start_date = btc_halving['date'][i]
        
    end_date = btc_halving['date'][i+1]
    mask = (data.index >= start_date) & (data.index < end_date)
        
    # Rellenar columna 'reward' con las recompensas de minería
    data.loc[mask, 'reward'] = btc_halving['reward'][i]
    
    # Rellenar columna 'countdown_halving' con los días restantes
    time_to_next_halving = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    
    data.loc[mask, 'countdown_halving'] = np.arange(time_to_next_halving.days)[::-1][:mask.sum()]

# Comprobar que se han creado los datos correctamente
# ==============================================================================
print('Segundo halving:', btc_halving['date'][2])
print(data.loc['2016-07-08':'2016-07-09'])
print('')
print('Tercer halving:', btc_halving['date'][3])
print(data.loc['2020-05-10':'2020-05-11'])
print('')
print('Próximo halving:', btc_halving['date'][4])
data.tail(2)

# Gráfico de velas japonesas interactivo con Plotly
# ==============================================================================
candlestick = go.Candlestick(
                  x     = data.index,
                  open  = data.open,
                  close = data.close,
                  low   = data.low,
                  high  = data.high,
              )

fig = go.Figure(data=[candlestick])

fig.update_layout(
    width       = 1200,
    height      = 600,
    title       = dict(text='<b>Chart Bitcoin/USD</b>', font=dict(size=30)),
    yaxis_title = dict(text='Precio (USD)', font=dict(size=15)),
    margin      = dict(l=10, r=20, t=80, b=20),
    shapes      = [dict(x0=btc_halving['date'][2], x1=btc_halving['date'][2], 
                        y0=0, y1=1, xref='x', yref='paper', line_width=2),
                   dict(x0=btc_halving['date'][3], x1=btc_halving['date'][3], 
                        y0=0, y1=1, xref='x', yref='paper', line_width=2),
                   dict(x0=btc_halving['date'][4], x1=btc_halving['date'][4], 
                        y0=0, y1=1, xref='x', yref='paper', line_width=2)
                  ],
    annotations = [dict(x=btc_halving['date'][2], y=1, xref='x', yref='paper',
                      showarrow=False, xanchor='left', text='Segundo halving'),
                   dict(x=btc_halving['date'][3], y=1, xref='x', yref='paper',
                      showarrow=False, xanchor='left', text='Tercer halving'),
                   dict(x=btc_halving['date'][4], y=1, xref='x', yref='paper',
                      showarrow=False, xanchor='left', text='Cuarto halving')
                  ],
    xaxis_rangeslider_visible = False,
)

# Guarda el gráfico en un archivo HTML local
pio.write_html(fig, 'plot.html', auto_open=True)