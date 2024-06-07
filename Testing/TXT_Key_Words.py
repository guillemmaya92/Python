import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Cargar las reseñas en un DataFrame
data = {
    'reviews': [
        "El producto es excelente y me encanta. excelent",
        "Mala atención al cliente, no volveré a comprar excelente.",
        "La calidad es buena, pero el precio es muy alto.",
        "Estoy satisfecho con la compra, la entrega fue rápida.",
        "El artículo llegó dañado y el servicio fue terrible."
    ]
}
df = pd.DataFrame(data)

# 2. Preprocesamiento del texto
# Convertir a minúsculas
df['reviews'] = df['reviews'].str.lower()

# Tokenización y eliminación de stopwords
stop_words = set(stopwords.words('spanish'))
df['reviews'] = df['reviews'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# 3. Extraer palabras clave usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['reviews'])

# Convertir a un DataFrame para visualizar
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Sumar las puntuaciones TF-IDF de cada palabra
word_scores = tfidf_df.sum(axis=0)

# Ordenar las palabras por su importancia
word_scores = word_scores.sort_values(ascending=False)

# Mostrar las 10 palabras clave más importantes
print(word_scores.head(10))
