import streamlit as st
from transformers import pipeline

st.header("Análisis de Sentimientos y Traducción con Transformers 📝💬")

st.markdown("""
En esta página, utilizaremos la biblioteca **Transformers** para realizar un análisis de sentimientos y traducción de texto.

Utilizaremos la función `pipeline` de Transformers, que nos permite acceder a modelos pre-entrenados para diferentes tareas de procesamiento de lenguaje natural (NLP).

A continuación, ingresa una oración en español y veremos cómo el modelo analiza el sentimiento y realiza la traducción al inglés.

¡Diviértete! 😄🚀
""")

# Cargar las pipelines
sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
translation = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-es-en")

# Entrada de texto
user_input = st.text_input("Introduce una oración en español")

# Función para procesar el texto
def analyze_and_translate(sentence):
    output = ""

    # Análisis de sentimientos
    sentiment = sentiment_analysis(sentence)[0]
    output = output + f"Sentimiento: {sentiment['label']}, Score: {sentiment['score']}\n"

    # Traducción
    translated_text = translation(sentence)[0]['translation_text']
    output = output + f"Traducción: {translated_text}"

    return output

# Si se ingresa un texto, procesar las pipelines
if user_input:
    output = analyze_and_translate(user_input)
    st.text(output)
# Agregar creditos al final de sidebar Por Missael Barco
