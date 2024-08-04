# Libraries
# =====================================================================
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects

# Data Extraction (Countries)
# =====================================================================
# Extr
# Obtener el JSON desde la URL
url = "https://raw.githubusercontent.com/guillemmaya92/world_map/main/Dim_Country.json"
response = requests.get(url)
data = response.json()

# Convertir el JSON en un DataFrame y resetear el índice
dfc = pd.DataFrame.from_dict(data, orient='index').reset_index()
dfc.rename(columns={'index': 'ISO3'}, inplace=True)

# Data Extraction (IMF)
# =====================================================================
# Definir los parámetros
parameters = ["PPPPC", "LP"]

# Definir la URL base de la API
base_url = "https://www.imf.org/external/datamapper/api/v1/"

# Crear una lista para almacenar todos los registros
all_records = []

# Iterar sobre los parámetros y realizar solicitudes a la API
for param in parameters:
    url = f"{base_url}{param}"
    response = requests.get(url)
    data = response.json()  # Convertir la respuesta a JSON
    
    # Extraer los valores específicos
    values = data['values'][param]
    
    # Crear registros para cada parámetro
    for country, years in values.items():
        for year, value in years.items():
            all_records.append({"Parameter": param, "Country": country, "Year": int(year), "Value": value})

# Convertir los registros a un DataFrame de Pandas
df = pd.DataFrame(all_records)

# Data Manipulation
# =====================================================================
# Trasponer columnas
df = df.pivot_table(index=['Country', 'Year'], columns='Parameter', values='Value').reset_index()

# Merge consultas
df = df.merge(dfc, how='left', left_on='Country', right_on='ISO3')
df = df[['ISO3', 'Year', 'LP', 'PPPPC', 'Analytical', 'Region']]

# Filter nulls and order
df = df[(df['PPPPC'].notna()) & (df['LP'].notna()) & (df['Region'].notna()) ]
df = df.sort_values(by=['Year', 'PPPPC'])

# Calcular los acumulados de 'widths' para la columna 'left'
df['left'] = df.groupby('Year')['LP'].cumsum() - df['LP']

# Calcular el promedio ponderado por grupo
weighted_avg = df.groupby('Year').apply(lambda x: np.average(x['PPPPC'], weights=x['LP']))
weighted_avg = weighted_avg.reset_index(name='avg_weighted_score')

# Unir el promedio ponderado al DataFrame original
df = df.merge(weighted_avg, on='Year', how='left')

# Add a GDP column calculated
df['GDP'] = df['PPPPC'] * df['LP']
df['GDPsum'] = df.groupby('Year')['GDP'].cumsum()

# Renombrar columnas
df.rename(columns={
    'ISO3': 'Country',
    'Year': 'Year',
    'LP': 'Population',
    'PPPPC': 'PIBC',
    'Analytical': 'Analytical',
    'Region': 'Region',
    'left': 'Left',
    'avg_weighted_score': 'PIBAvg',
    'GDP': 'PIB',
    'GDPsum': 'PIBsum'
}, inplace=True)

# Data Visualization
# =====================================================================
# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Crear una paleta de colores
palette = sns.color_palette("coolwarm", as_cmap=True)

# Invertir la paleta de colores
palette = palette.reversed()

# Función para actualizar el gráfico
def update(year):
    plt.clf()  # Limpiar la figura actual
    subset = df[df['Year'] == year]
    
    # Normalizar los valores de PIBC para que estén en el rango [0, 1]
    norm = plt.Normalize(subset['PIBsum'].min(), subset[subset['Country'] == 'USA']['PIBsum'].values[0])
    colors = palette(norm(subset['PIBsum']))
    
    # Crear el gráfico de barras con Matplotlib usando los datos del año actual
    plt.bar(subset['Left'], subset['PIBC'], width=subset['Population'], 
            color=colors, alpha=1, align='edge', edgecolor='grey', linewidth=0.25)
    
    subsetusa = subset[subset['Country'] == 'USA']
    subset['Population_cumsum'] = subset['Population'].cumsum()
    
    plt.xlim(0, subset['Population_cumsum'].max())
    plt.ylim(0, subsetusa['PIBC'].max() * 1.05)
    plt.grid(axis='x')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Añadir etiquetas y título
    plt.xlabel('Cumulative Global Population (M)', fontsize=10, fontweight='bold')
    plt.ylabel('Real GDP per capita (US$)', fontsize=10, fontweight='bold')
    plt.title(f'Distribution of GDP per capita', fontsize=16, fontweight='bold', pad=20)
    
    # Calcular la mediana de PIBC
    subset_sorted = subset.sort_values(by='Left')  # Ordenar por la variable 'left' para cálculo correcto
    total_lp = subset_sorted['Population'].sum()
    cumulative_lp = subset_sorted['Population'].cumsum()
    median_idx = np.searchsorted(cumulative_lp, total_lp / 2.0)
    weighted_median = subset_sorted.iloc[median_idx]['PIBC']
    plt.axhline(y=weighted_median, color='darkred', linestyle='--', linewidth=0.5, label=f'Mediana: {weighted_median:,.0f}')
    
    plt.text(
        x=subset['Left'].max() * 0.1,
        y=weighted_median * 1.1,
        s=f'Median: {weighted_median:,.0f}',
        color='darkred',
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=10,
        weight='bold'
    )
    
    pibc_usa = subsetusa.iloc[0]['PIBC']
    plt.axhline(y=pibc_usa, color='darkblue', linestyle='--', linewidth=0.5, label=f'GDP USA: {pibc_usa:,.0f}')
    
    plt.text(
        x=subset['Left'].max() * 0.1,
        y=pibc_usa * 0.95,
        s=f'USA: {pibc_usa:,.0f}',  # Usamos el valor del PIBC para 'USA'
        color='darkblue',
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=10,
        weight='bold'
    )

    plt.gca().text(
                subset['Left'].max() * 0.07,
                pibc_usa * 0.93,
                f'vs Median: {pibc_usa / weighted_median * 100:,.0f}%', 
                ha='center', va='center', 
                fontsize=9, 
                color='darkblue') 
    
    # Añadir el año en la parte superior derecha del gráfico
    plt.text(0.95, 1.06, f'{year}',
             transform=plt.gca().transAxes,
             fontsize=22, ha='right', va='top',
             fontweight='bold', color='#D3D3D3')

# Configurar la animación
years = df['Year'].unique()
fig = plt.figure(figsize=(16, 9))
ani = animation.FuncAnimation(fig, update, frames=years, repeat=False)

# Guardar la animación
ani.save('C:/Users/guill/Downloads/CountriesX.webp', writer='imagemagick', fps=4)

# Mostrar el gráfico
plt.show()