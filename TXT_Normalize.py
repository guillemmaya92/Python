from fuzzywuzzy import process

def normalizar_busqueda(busqueda, umbral):
    # Procesar la coincidencia aproximada con las propias búsquedas
    mejor_coincidencia, score = process.extractOne(busqueda, busquedas)
    
    # Si el puntaje de coincidencia no alcanza el umbral deseado, normalizar utilizando una parte de la cadena de búsqueda
    if score < umbral:
        # Puedes implementar una estrategia más sofisticada aquí, por ejemplo, tomar las primeras palabras de la búsqueda
        return busqueda.split()[0]  # En este caso, solo tomamos la primera palabra de la búsqueda
    else:
        return mejor_coincidencia

# Ejemplos de términos de búsqueda
busquedas = ["pbi", "power bi", "powwer bi", "Power BAI", "microsoft power bi", "pewer bi", "pwower bi", "excel", "excelo", "excelente", "qlq", "qlik sense", "qlikview", "spot fire", "spot fyre", "spotfyre"]

# Definir un umbral para la coincidencia aproximada
umbral = 80

# Normalizar cada búsqueda
busquedas_normalizadas = [normalizar_busqueda(busqueda, umbral) for busqueda in busquedas]

# Imprimir los resultados
for busqueda, busqueda_normalizada in zip(busquedas, busquedas_normalizadas):
    print(f"Original: {busqueda}, Normalizado: {busqueda_normalizada}")