import pandas as pd
from geopy.geocoders import Nominatim

# Crear instancia del geolocalizador
geolocator = Nominatim(user_agent="my_user_agent")

# Definir la ciudad y el país
city = "SUMINISTROS VIDAL"
country = "Spain"

# Obtener la ubicación (latitud y longitud)
loc = geolocator.geocode(city + ', ' + country)

# Crear un DataFrame con la latitud y longitud
data = {'City': [city], 'Country': [country], 'Latitude': [loc.latitude], 'Longitude': [loc.longitude]}
df = pd.DataFrame(data)

# Imprimir el DataFrame
print(df)
