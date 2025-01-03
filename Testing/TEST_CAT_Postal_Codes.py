# Libraries
# =================================
import requests
import xml.etree.ElementTree as ET
import pandas as pd

# Extract Data (Regions)
# =================================
# URL del XML
url = "https://analisi.transparenciacatalunya.cat/api/views/9aju-tpwc/rows.xml?accessType=DOWNLOAD"

# Get XML file
response = requests.get(url)
xml_content = response.content
    
# Parse XML
root = ET.fromstring(xml_content)
 
# Extract data and append it
data = []
for row in root.findall(".//row"):
    row_data = {}
    for element in row:
        row_data[element.tag] = element.text
    data.append(row_data)
    
# Create dataframe
dfmc = pd.DataFrame(data)

# Filter nulls and select columns
dfmc = dfmc.dropna(how="all")
dfmc = dfmc[["codi", "nom", "codi_comarca", "nom_comarca", "latitud", "longitud"]]

# Extract Data (Postal Codes)
# =================================
# URL del XML
url = "https://analisi.transparenciacatalunya.cat/api/views/tp8v-a58g/rows.xml?accessType=DOWNLOAD"

# Get XML file
response = requests.get(url)
xml_content = response.content
    
# Parse XML
root = ET.fromstring(xml_content)
 
# Extract data and append it
data = []
for row in root.findall(".//row"):
    row_data = {}
    for element in row:
        row_data[element.tag] = element.text
    data.append(row_data)
    
# Create dataframe
dfcp = pd.DataFrame(data)

# Filter nulls and select columns
dfcp = dfcp.dropna(how="all")
dfcp = dfcp[["codi_postal", "codi_municipi", "nom_municipi"]]

# Transformation Data
# ================================= 
# Merge dataframes
df = pd.merge(dfmc, dfcp, left_on='codi', right_on='codi_municipi', how='left')   

# Remove postal code duplicates
df = df.drop_duplicates(subset="codi_postal", keep='last')

# Select columns
df = df[["codi_postal", "codi_municipi", "nom_municipi", "codi_comarca", "nom_comarca", "latitud", "longitud"]]

# Export to csv
df.to_csv('C:/Users/guillem.maya/Downloads/postal_codes.csv', index=False, encoding='utf-8')
df.to_excel('C:/Users/guillem.maya/Downloads/postal_codes.xlsx', index=False)

# Mostrar el DataFrame
print(df)
