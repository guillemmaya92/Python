import pyodbc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Conexión SQL Server
# ==============================================================================
server = ''  # Nombre o dirección IP del servidor
database = ''  # Nombre de la base de datos
connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
conn = pyodbc.connect(connection_string)
query = "SELECT * FROM H_Currencies WHERE symbol = 'EURUSD'"
df = pd.read_sql_query(query, conn)
conn.close()

# DataSet1
# ==============================================================================
# Definir y ordenar dataset
df['date'] = pd.to_datetime(df['date'])
df['year-month'] = df['date'].dt.to_period('M')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name()
df['day_of_week_num'] = df['date'].dt.dayofweek
df['day_of_week_name'] = df['date'].dt.day_name()
df = df.sort_values('date')


# Probability
# ==============================================================================
# Define parameters
Effect = 'Negative' #Positive/Negative
ConsecutiveDays = 2
Extra = 1

# Calculate consecutive changes
if Effect == 'Positive':
    df['consecutive'] = (df['changepercent'] < 0).astype(int)
else:
    df['consecutive'] = (df['changepercent'] > 0).astype(int)
df['consecutive'] = df['consecutive'].groupby((df['consecutive'] != df['consecutive'].shift()).cumsum()).cumsum()

# Filter and count consecutive rows
filterx = df[df['consecutive'] == ConsecutiveDays]
countx = len(filterx)

# Filter and count consecutive rows + n
filtery = df[df['consecutive'] == ConsecutiveDays + Extra]
county = len(filtery)

# Calculate the probability
probability = county / countx if county != 0 else 0

print(f"The probability of going {'up' if Effect == 'Positive' else 'down'} {Extra} day more after {ConsecutiveDays} consecutive {'positive' if Effect == 'Positive' else 'negative'} days is: {probability}")