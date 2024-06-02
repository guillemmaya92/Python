# Data processing
# ==============================================================================
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Date

# Parameters
# ==============================================================================
p_from = '1900-01-01'
p_to = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
apikey = "0RU79WomyvojlRcstiO0IpgAg1O4aIuA"

getdates = 'range' #(today/ range)
getvalues = 'crypto' #(crypto / forex)
getall = False #(True, False)

# Get Values
# ==============================================================================
if getvalues == 'crypto' and getall:
    # Get crypto
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 150, 'page': 1, 'sparkline': False}
    response = requests.get(url, params=params)
    data = response.json()
    coin_list = [f"{crypto['symbol'].upper()}USD" for crypto in data]
elif getvalues == "forex" and getall:
    # Get forex
    url = f"https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs?apikey={apikey}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[df['currency'] == 'USD']
    coin_list = df['symbol'].tolist()
    coin_list.append('BTCUSD')
else:
    coin_list = ["EURUSD"]

# API data extraction
# ==============================================================================
# List for store dataframes of each exchange
dfs = []

# Iterate over each exchange list
for i in coin_list:
    # Get URL data
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{i}?from={p_from}&to={p_to}&apikey={apikey}"
    response = requests.get(url)
    data = response.json()

    # Check if there is data
    if 'historical' not in data:
        continue

    # Navigate and convert to DataFrame
    navigate = data['historical']
    df = pd.DataFrame(navigate)
    df['symbol'] = i

    # Set index and fill empty
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.asfreq('D').ffill()
    df = df.reset_index()

    # Add moving average columns
    df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['ma100'] = df['close'].rolling(window=100, min_periods=1).mean()
    df['ma200'] = df['close'].rolling(window=200, min_periods=1).mean()
    df['ma300'] = df['close'].rolling(window=300, min_periods=1).mean()
    df['open'] = df.groupby('symbol')['close'].shift(1).fillna(df['open'])
    df['change'] = df['close'] - df['open']
    df['changepercent'] = (df['close'] - df['open']) / df['open']
    df['changesign'] = df['change'].apply(lambda x: '+' if x > 0 else '-' if x < 0 else '=')

    # Select and rename columns
    df = df[['symbol', 'date', 'open', 'close', 'low', 'high', 'volume', 'change', 'changepercent', 'changesign', 'ma10', 'ma20', 'ma50', 'ma100', 'ma200', 'ma300']]
    df.columns = ['symbol', 'date', 'open', 'close', 'low', 'high', 'volume', 'change', 'changepercent', 'changesign', 'ma10', 'ma20', 'ma50', 'ma100', 'ma200', 'ma300']

    # Format columns
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['open'] = df['open'].astype(float).round(5)
    df['close'] = df['close'].astype(float).round(5)
    df['low'] = df['low'].astype(float).round(5)
    df['high'] = df['high'].astype(float).round(5)
    df['volume'] = df['volume'].astype(float).round(0)
    df['change'] = df['change'].astype(float).round(5)
    df['changepercent'] = df['changepercent'].astype(float).round(5)
    df['ma10'] = df['ma10'].astype(float).round(5)
    df['ma20'] = df['ma20'].astype(float).round(5)
    df['ma50'] = df['ma50'].astype(float).round(5)
    df['ma100'] = df['ma100'].astype(float).round(5)
    df['ma200'] = df['ma200'].astype(float).round(5)
    df['ma300'] = df['ma300'].astype(float).round(5)

    # Conditional filter for today or range dataset
    if getdates == "today":
        df = df[df['date'] == datetime.strptime(p_to, '%Y-%m-%d').date()]

    # Add datafram to list
    dfs.append(df)

# Concatenate all DataFrames into one
if dfs:  # Check if dfs list is not empty
    df = pd.concat(dfs, ignore_index=True)
else:
    df = pd.DataFrame(columns=['symbol', 'date', 'open', 'close', 'low', 'high', 'volume', 'change', 'changepercent', 'changesign', 'ma10', 'ma20', 'ma50', 'ma100', 'ma200', 'ma300'])

# Sorting by date ascending
df = df.sort_values(by='date', ascending=True)

# SQL Server connection
# ==============================================================================
# SQL Server connection details
server = 'localhost'
database = 'database'
table_name = 'H_Forex' if getvalues == 'forex' else 'H_Crypto'
    
# Create a connection string using username and password
engine_url = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
engine = create_engine(engine_url)
metadata = MetaData()

# SQL Server Dumping Data
# ==============================================================================
# Define the table structure
exchange_table = Table(
    table_name, metadata,
    Column('symbol', String(10)),
    Column('date', Date),
    Column('open', Float),
    Column('close', Float),
    Column('low', Float),
    Column('high', Float),
    Column('change', Float),
    Column('ma10', Float),
    Column('ma20', Float),
    Column('ma50', Float),
    Column('ma100', Float),
    Column('ma200', Float),
    Column('ma300', Float)
)

# Create the table if it doesn't exist
metadata.create_all(engine)

# Insert DataFrame to SQL Server
with engine.connect() as connection:
    df.to_sql(table_name, con=connection, if_exists='replace', index=False)

# Show result
print(df)
