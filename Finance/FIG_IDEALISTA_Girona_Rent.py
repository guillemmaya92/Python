# Libraries
# ===================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Extract Data
# ===================================================
# Ruta del archivo CSV
file_path = r'C:\Users\guill\Downloads\girones_lloguer.csv'

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path, sep=';', encoding='latin1')

# Transform Data
# ===================================================
# Define max price and cheap price
minprice = 300
maxprice = 2000
cheapprice = 800

# Filter outliers and define price
df = df[df['price'] > minprice]
df = df[df['price'] < maxprice]

medianprice = df['price'].median()

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
sns.swarmplot(x='price', data=df, hue='color', palette={'cheap': '#ffc939', 'median': '#a8c2d2', 'expensive': '#477794'}, orient="h", alpha=0.7)

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