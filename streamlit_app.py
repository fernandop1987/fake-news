import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scikit-learn.feature_extraction.text import TfidfVectorizer
from scikit-learn.model_selection import train_test_split
from scikit-learn.linear_model import LogisticRegression
from scikit-learn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from scikit-learn.metrics import confusion_matrix
from scikit-learn.metrics import classification_report

#######################
# Load data

# Cargar el archivo CSV usando pandas
data = pd.read_excel('dataset.xlsx')

#######################
# Genero el modelo

# merging the author name and news title
data['contenido'] = data['Fuente']+' '+data['Titular']

# Separamos la variable dependiente e independientes

## Variables independientes
X = data.drop(columns='Etiqueta', axis=1)

## Variable independiente
Y = data['Etiqueta']

# Stemming
port_stem = PorterStemmer()
def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('spanish')]
    review = ' '.join(review)
    return review

data['contenido'] = data['contenido'].apply(stemming)

# Separamos data y etiqueta

X = data['contenido'].values
Y = data['Etiqueta'].values


# TF-IDF

# Convertimos el texto a variables numéricas
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

#Split Train - Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#Regresión Logística
model = LogisticRegression()
model.fit(X_train, Y_train)

#######################

# Cargar el modelo y el vectorizador guardados
##model = joblib.load('logistic_regression_model.pkl')
##vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Función para predecir si una noticia es fake o no
def predict_fake_news(source, headline):
    # Combinar fuente y titular
    content = source + ' ' + headline
    
    # Preprocesar el contenido
    preprocessed_content = stemming(content)
    
    # Transformar el texto con el vectorizador TF-IDF
    content_vectorized = vectorizer.transform([preprocessed_content])
    
    # Hacer la predicción
    prediction = model.predict(content_vectorized)
    
    return prediction[0]

# Configurar la aplicación en Streamlit
st.title('Detección de Fake News')

# Crear entradas de texto para la fuente y el titular
source = st.text_input('Fuente de la noticia')
headline = st.text_input('Titular de la noticia')

# Botón para hacer la predicción
if st.button('Verificar'):
    if source and headline:
        prediction = predict_fake_news(source, headline)
        
        if prediction == 1:
            st.success('La noticia es verdadera.')
        else:
            st.error('La noticia es falsa.')
    else:
        st.warning('Por favor, ingrese tanto la fuente como el titular.')
