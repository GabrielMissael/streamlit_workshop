import streamlit as st

st.set_page_config(
    page_title="Hola",
    page_icon="👋",
)

# Agreaga logo en sidebar
st.sidebar.image("examples/logo.png", width=150)

st.sidebar.success("Selecciona una demo 📝")

# Agreagar creditos al final de sidebar Por Missael Barco
st.sidebar.markdown("""
    ---
    *Creado con 💖 por [Missael Barco](https://www.linkedin.com/in/gmissaelbarco/) | [GitHub](https://github.com/GabrielMissael)*
""")

st.title("Aplicación Interactiva con Streamlit 💻🌐")
st.markdown("""
    Esta aplicación web incluye tres proyectos diferentes que utilizan Machine Learning:

    1. Clasificación de dígitos con SVMs: Puedes subir una imagen de un dígito y el modelo SVM clasificará el dígito.
    2. Análisis de Sentimientos y Traducción con Transformers: Introduce un texto en español y el modelo te dará el análisis de sentimientos y su traducción al inglés.
    3. Detección de Objetos con YOLOv8 Nano: Sube una imagen y YOLOv8 Nano detectará los objetos presentes en la imagen.

    Usa el menú lateral para navegar entre los proyectos. ¡Disfrútalo!
""")