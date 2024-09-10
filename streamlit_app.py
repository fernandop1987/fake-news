import streamlit as st
import spacy
import joblib

# Cargar el modelo y el vectorizador guardados
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Cargar el lematizador en español de Spacy
nlp = spacy.load('es_core_news_sm')

# Función de lematización
def lemmatization(content):
    doc = nlp(content)
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(lemmas)

# Función para predecir si una noticia es fake o no
def predict_fake_news(source, headline):
    # Combinar fuente y titular
    content = source + ' ' + headline
    
    # Preprocesar el contenido
    preprocessed_content = lemmatization(content)
    
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
