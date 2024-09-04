import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# Income Inequality (2022)
# =====================================================
# Datos de entrada
x = np.array([0, 0.5, 0.9, 0.99, 1])
y = np.array([0, 0.2008, 0.6687, 0.8998, 1])

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 100)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfi = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Income Inequality (1980)
# =====================================================
# Datos de entrada
x = np.array([0, 0.5, 0.9, 0.99, 1])
y = np.array([0, 0.2313, 0.6367, 0.8834, 1])

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 100)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfi80 = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Wealth Inequality (2022)
# =====================================================
# Datos de entrada
x = np.array([0, 0.5, 0.9, 0.99, 1])
y = np.array([0, 0.0682, 0.4341, 0.7722, 1])

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 100)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfw = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# Wealth Inequality (1985)
# =====================================================
# Datos de entrada
x = np.array([0, 0.5, 0.9, 0.99, 1])
y = np.array([0, 0.0435, 0.3463, 0.7581, 1])

# Crear un interpolador monotónico (PCHIP)
interpolador = PchipInterpolator(x, y)

# Generar valores suaves para x
x_suave = np.linspace(min(x), max(x), 100)
y_suave = interpolador(x_suave)

# Crear un DataFrame con los valores interpolados
dfw80 = pd.DataFrame({
    'x_suave': x_suave,
    'y_suave': y_suave
})

# DataFrame Equality
# =====================================================================
data = {
    'POP_Cum': [0, 1],
    'GDP_Cum': [0, 1]
}

dfe = pd.DataFrame(data)

# Data Visualization
# =====================================================
# Font Style
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Open Sans'], 'font.size': 10})

# Create figure and lines
plt.figure(figsize=(10, 10))
plt.plot(dfi['x_suave'], dfi['y_suave'], label='Income (2022)', color='darkblue')
plt.plot(dfi80['x_suave'], dfi80['y_suave'], label='Income (1980)', color='darkblue', linewidth=0.5, linestyle='-')
plt.plot(dfw['x_suave'], dfw['y_suave'], label='Wealth (2022)', color='darkred')
plt.plot(dfw80['x_suave'], dfw80['y_suave'], label='Wealth (1980)', color='darkred', linewidth=0.5, linestyle='-')
plt.plot(dfe['POP_Cum'], dfe['GDP_Cum'], label='Perfect Distribution', color='darkgrey')

# Title and labels
plt.suptitle('   Spain Inequality 1980-2022', fontsize=16, fontweight='bold', y=0.95)
plt.title('Income and Wealth distribution', fontsize=12, fontweight='bold', color='darkgrey', pad=20)
plt.xlabel('Cumulative Population (%)', fontsize=10, fontweight='bold')
plt.ylabel('Cumulative Income/Wealth (%)', fontsize=10, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()

# Configuration
plt.grid(True, linestyle='-', color='grey', linewidth=0.08)
plt.gca().set_aspect('equal', adjustable='box')

# Save the figure
plt.savefig('C:/Users/guillem.maya/Desktop/FIG_GINI_Income_Wealth.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()