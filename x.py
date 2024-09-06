import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# Define CSV path
path = 'C:\\Users\\guillem.maya\\Downloads\\data'

# List to save dataframe
list = []

# Iterate over each file
for archivo in os.listdir(path):
    if archivo.startswith("WID_data_") and archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, archivo), delimiter=';')
        list.append(df)

# Combina todos los DataFrames en uno solo
df = pd.concat(list, ignore_index=True)
dfv = df.copy()

# Filter dataframe
country = ['US', 'ES']
variable = ['shwealj992', 'scaincj992', 'sdiincj992']
variablev = ['adiincj992', 'ahwealj992']
percentile = ['p10p100', 'p20p100', 'p30p100', 'p40p100', 'p50p100', 'p60p100', 'p70p100', 'p80p100', 'p90p100']
percentilev = ['p0p100']
year = [1985, 2022]
df = df[df['variable'].isin(variable) & df['percentile'].isin(percentile) & df['year'].isin(year)]
dfv = dfv[dfv['variable'].isin(variablev) & dfv['percentile'].isin(percentilev) & dfv['year'].isin(year)]

# Transformation
df['value'] = 1 - df['value']
df['percentile'] = df['percentile'].str[1:3].astype(int) / 100
df = df[['country', 'variable', 'year', 'percentile', 'value']]

# Transformation 2
dfv = dfv.pivot_table(index=['country', 'percentile', 'year'], 
                          columns='variable', values='value').reset_index()

# Add values 0 and 1
dfx = pd.DataFrame(
    [(c, v, y, p, p) for c in country for v in variable for y in year for p in [0, 1]],
    columns=['country', 'variable', 'year', 'percentile', 'value']
)
df = pd.concat([df, dfx], ignore_index=True)
df = df.sort_values(by=['country', 'variable', 'year', 'percentile']).reset_index(drop=True)

# ==============
# Crear una función para realizar la interpolación
def aplicar_interpolacion(sub_df):
    x = sub_df['percentile'].values
    y = sub_df['value'].values
    interpolator = PchipInterpolator(x, y)
    
    # Puedes usar el interpolador para nuevos valores de x o evaluar en los existentes
    # Aquí solo para ejemplo usamos el rango original de x
    x_smooth = np.linspace(min(x), max(x), num=100)  # Generar nuevos puntos
    y_smooth = interpolator(x_smooth)
    
    # Devolvemos un DataFrame con los resultados interpolados
    return pd.DataFrame({
        'country': sub_df['country'].iloc[0],
        'variable': sub_df['variable'].iloc[0],
        'year': sub_df['year'].iloc[0],
        'percentile': x_smooth,
        'value': y_smooth
    })

# Aplicar la función a cada grupo de country, variable, y year
df = df.groupby(['country', 'variable', 'year']).apply(aplicar_interpolacion).reset_index(drop=True)

# Ver los resultados
print(dfv)