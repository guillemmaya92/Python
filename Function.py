import pandas as pd
import numpy as np

def calcular_columnas(df):
  
    # Calcula las columnas necesarias
    dfc = df.copy()
    dfc['Recuento'] = dfc.index + 1
    dfc['Acumulativo'] = dfc['Recuento'] + dfc['Recuento'].shift(1).fillna(0)
    dfc['Suma'] = dfc['Acumulativo'].sum()
    dfc['Division'] = dfc['Acumulativo'] / dfc['Suma']
    dfc['Multiplicacion'] = dfc['Inicial'] * dfc['Division']
    dfc['Prorrateado'] = dfc['Multiplicacion'] * len(dfc)
    df['PPPPC'] = dfc['Prorrateado']
    return df

# Ejemplo de uso
data = pd.DataFrame({
    'Pais': ['Pais1'] * 50 + ['Pais2'] * 50,
    'Ano': [2019, 2020, 2021, 2022, 2023] * 10 * 2,
    'Inicial': np.random.rand(100).tolist()
})

data_calculada = data.groupby(['Pais', 'Ano']).apply(calcular_columnas).reset_index(drop=True)
print(data_calculada)