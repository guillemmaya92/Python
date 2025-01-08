# Libraries
# ===================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Extract Data
# ===================================================
# URL Github
url = "https://raw.githubusercontent.com/guillemmaya92/Python/main/Data/Catalunya_CP.csv"

# Read in a dataframe
df = pd.read_csv(url, encoding='latin1', sep=';')

# Select columns
df = df[['province', 'region', 'price']]

# Filter region
df = df[df['region'].str.lower() == 'barcelonès']

# Transform Data
# ===================================================
# Define max price and cheap price
minprice = 300
maxprice = 2000
cheapprice = 800

# Filter outliers and define price
df = df[df['price'] > minprice]
df = df[df['price'] < maxprice]

# Calculate median price
medianprice = df['price'].median()

# Function to distribute values
def add_perfect_jitter(series, jitter_range=100):
    jittered_values = []
    for value in series.unique():
        # Find the position with dots with same values
        indices = series[series == value].index
        num_points = len(indices)
        
        # Create a uniformly distributed offset of those points
        jitter_values_for_value = np.linspace(-jitter_range/2, jitter_range/2, num_points)
        
        # Assign the displaced values to the corresponding positions
        jittered_values.extend(jitter_values_for_value)
    
    # Devuelve la lista de valores desplazados
    return np.array(jittered_values)

# Aplicar el desplazamiento perfecto a la columna 'price'
df['jitter'] = add_perfect_jitter(df['price'], jitter_range=100)
df['price'] = np.where(df['price'] < 800, df['price'], df['price'] + df['jitter'])

# Round values in a ranges
df['price'] = df['price'].apply(lambda x: round(x / 12) * 12)

# Classification
dfsorted = np.sort(df['price'])
medianrow = len(dfsorted) // 2  # Tamaño de la mitad más pequeña
dfhalf = dfsorted[:medianrow]  # Obtener la mitad más pequeña

# Apply color classification
df['color'] = df['price'].apply(
    lambda x: 'cheap' if x < cheapprice 
              else ('median' if x <= medianprice and x in dfhalf 
                    else 'expensive')
)

# Mostrar el DataFrame
print(df)

# Visualize Data
# ===================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Named values
num_cheap = len(df[df['color'] == 'cheap'])
num_total = len(df)
num_median = int(len(df) / 2)

# Create figure and swarmplot
plt.figure(figsize=(10, 4))
sns.swarmplot(x='price', data=df, hue='color', palette={'cheap': '#ffc939', 'median': '#a8c2d2', 'expensive': '#477794'}, orient="h", alpha=0.7, size=3)

# Labels and legend
plt.text(0, 1.1, 'Pisos ofertados en Idealista por menos de 2.000 euros', fontsize=13, fontweight='bold', ha='left', transform=plt.gca().transAxes)
plt.text(0, 1.04, 'Anuncios en la província de Girona', fontsize=9, color='#262626', ha='left', transform=plt.gca().transAxes)
plt.xlabel('Precio (€)', color= '#4d4d4d', fontweight='bold')
plt.yticks([])
plt.legend([], [], frameon=False)

# Configuration
plt.axvline(x=cheapprice, color='grey', linestyle='--', linewidth=0.5, dashes=(5, 10))
plt.axvline(x=medianprice, color='grey', linestyle='--', linewidth=0.5, dashes=(5, 10)) 
plt.axhline(y=0, color='lightgrey', linestyle='-', linewidth=0.5)
plt.xlim(minprice * 1.3, maxprice * 1.01 )
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_color('grey')
plt.gca().spines['bottom'].set_linewidth(0.7)
plt.tick_params(axis='x', colors='grey')

# Add text with info
plt.text(minprice * 1.35, -0.05, f'Girona', 
         fontsize=10, fontweight='bold', color='black', rotation=0)
plt.text(minprice * 1.35, 0.12, f'Total: {num_total}\nanuncios', 
         fontsize=10, fontweight='normal', color='black', ha='left', rotation=0)

# Add cheap info
plt.text((minprice + cheapprice)/2, -0.43, f'{num_cheap} pisos', 
         fontsize=8, fontweight='bold', color='black', ha='center', va='center', rotation=0,
         bbox=dict(facecolor='#a68221', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'))
plt.text((minprice + cheapprice)/2, -0.35, f"entre \n{int(minprice+100):,}€ y {int(cheapprice):,}€".replace(",", "."), 
         fontsize=7, fontweight='normal', color='#a68221', ha='center', va='center', rotation=0)

# Add median info
plt.text((medianprice + cheapprice)/2, -0.43, f'{num_median-num_cheap} pisos', 
         fontsize=8, fontweight='bold', color='black', ha='center', va='center', rotation=0,
         bbox=dict(facecolor='grey', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'))
plt.text((medianprice + cheapprice)/2, -0.35, f"entre \n{int(cheapprice):,}€ y {int(medianprice):,}€".replace(",", "."), 
         fontsize=7, fontweight='normal', color='grey', ha='center', va='center', rotation=0)

# Add expensive info
plt.text((medianprice + maxprice)/2, -0.43, f'{num_median+1} pisos', 
         fontsize=8, fontweight='bold', color='black', ha='center', va='center', rotation=0,
         bbox=dict(facecolor='#477794', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3'))
plt.text((medianprice + maxprice)/2, -0.35,  f"entre \n{int(medianprice):,}€ y {int(maxprice):,}€".replace(",", "."), 
         fontsize=7, fontweight='normal', color='#477794', ha='center', va='center', rotation=0)

# Add Data Source
plt.text(0, -0.15, 'Fuente: Idealista.', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    fontweight='bold',
    color='gray')

# Add Data Source
plt.text(0, -0.2, 'Notas: Cada bola representa un anuncio en la plataforma.', 
    transform=plt.gca().transAxes, 
    fontsize=8, 
    color='gray')

# Save the figure
plt.savefig('C:/Users/guill/Desktop/FIG_Idealista_Girona_Rent.png', format='png', dpi=300, bbox_inches='tight')

# Plot it!
plt.show()