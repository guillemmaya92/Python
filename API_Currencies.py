# Data processing
# ==============================================================================
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Date

# Parameters
# ==============================================================================
start = '1900-01-01'
end = '2024-05-26'
today = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
apikey = "YOURAPIKEY"

getdates = 'today' #(today/ range)
getvalues = 'forex' #(crypto / forex)
getall = False #(True, False)

p_from = today if getdates == 'today' else start
p_to = today if getdates == 'today' else end

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
    coin_list = ["BTCUSD"]

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

    # Select and rename columns
    df = df[['symbol', 'date', 'open', 'close', 'low', 'high', 'change', 'changePercent']]
    df.columns = ['symbol', 'date', 'open', 'close', 'low', 'high', 'change', 'changepercent']

    # Format columns
    df['date'] = pd.to_datetime(df['date'])
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['low'] = df['low'].astype(float)
    df['high'] = df['high'].astype(float)
    df['change'] = df['change'].astype(float)
    df['changepercent'] = df['changepercent'].astype(float)

    # Add datafram to list
    dfs.append(df)

# Concatenate all DataFrames into one
if dfs:  # Check if dfs list is not empty
    df = pd.concat(dfs, ignore_index=True)
else:
    df = pd.DataFrame(columns=['symbol', 'date', 'open', 'close', 'low', 'high', 'change', 'changepercent'])

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
    Column('changepercent', Float)
)

# Create the table if it doesn't exist
metadata.create_all(engine)

# Insert DataFrame to SQL Server
with engine.connect() as connection:
    df.to_sql(table_name, con=connection, if_exists='append', index=False)

# Show result
print(df)
