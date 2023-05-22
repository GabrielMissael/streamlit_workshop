import streamlit as st
import ultralytics
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import urllib

def bounding_boxes(img):
    model = YOLO("yolov8n.pt")

    model_output = model(img)[0]

    results = model_output.boxes.boxes.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    # Plot boxes
    for result in results:
        # Draw square
        ax.add_patch(plt.Rectangle((result[0], result[1]),
            result[2] - result[0],
            result[3] - result[1],
            fill=False,
            edgecolor='red',
            linewidth=2))

        # Get name
        name = model_output.names[int(result[5])]

        # Draw label
        ax.text(result[0], result[1] - 2,
            s=str(name) + ' ' + str(round(result[4], 2)),
            color='white',
            verticalalignment='top',
            bbox={'color': 'red', 'pad': 0})

    #Remove axes
    ax.axis('off')

    return fig

ultralytics.checks()

st.markdown('''

# Detección de Objetos con YOLOv8 Nano 🕵️‍♂️🖼️

### Tigre Hacks 2023 - Monterrey 🐯
-----

*Por: [Missael Barco](https://www.linkedin.com/in/gmissaelbarco/)*

En este notebook, vamos a explorar la detección de objetos en imágenes utilizando YOLOv8 Nano. YOLO (You Only Look Once) es una popular red neuronal de detección de objetos en tiempo real. YOLOv8 Nano es una versión reducida de la última versión de YOLO, que ha sido optimizada para funcionar en dispositivos con recursos computacionales limitados.

Esto nos permite utilizar el modelo en aplicaciones web interactivas, como las que podemos construir con Streamlit, sin que la detección de objetos ralentice significativamente la experiencia del usuario.

Durante este taller, vamos a cargar una imagen, pasarla por el modelo YOLOv8 Nano, y visualizar los objetos detectados en la imagen. Esto nos servirá como base para crear una aplicación Streamlit donde los usuarios puedan subir sus propias imágenes y ver los objetos que el modelo detecta.

Recuerda que este es solo un vistazo a lo que puedes lograr con la detección de objetos y los modelos pre-entrenados. ¡El cielo es el límite cuando se trata de aplicaciones de Computer Vision!
''')

picture = st.camera_input("Tomá una foto de lo que quieras clasificar 📸")

if picture is not None:
    img = PIL.Image.open(picture)

processed_picture = bounding_boxes(img=img)

st.pyplot(processed_picture)
