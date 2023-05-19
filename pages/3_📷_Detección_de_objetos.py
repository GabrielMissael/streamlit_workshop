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

img = None

st.markdown('''
# YOLOv8n Demo - HackCIMAT 👑
------

¡Hola! En esta página web puedes probar el modelo YOLOv8n entrenado para detectar objetos en imágenes.
Toda esta aplicación tiene menos de 100 líneas de código, y está hecha con [Streamlit](https://streamlit.io/).
Puedes revisar el código en [GitHub](https://github.com/GabrielMissael/CV_YOLO_workshop)

Puedes probarlo con imágenes de internet, subir una imagen desde tu computadora, o tomar una foto con tu cámara web.
''')

# Sidebar
st.sidebar.markdown('''
## Opciones
''')

# Select image source
source = st.sidebar.selectbox("¿De dónde quieres cargar la imagen?", ("Internet", "Subir imagen", "Cámara web"))

# Load image
if source == "Internet":
    url = st.sidebar.text_input("URL de la imagen", "")

    if url != "":
        try:
            img = PIL.Image.open(urllib.request.urlopen(url))
        except:
            st.sidebar.error("No se pudo cargar la imagen")
            img = None

elif source == "Subir imagen":
    img = st.sidebar.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])

    if img is not None:
        img = PIL.Image.open(img)

elif source == "Cámara web":
    img_photo = st.camera_input(label="Toma una foto 📷", key="camera")

    if img_photo is not None:
        img = PIL.Image.open(img_photo)

# Show image
if img is not None:
    if source != "Cámara web":
        st.image(img, caption='Imagen original', use_column_width=True)

    # Show bounding boxes
    st.pyplot(bounding_boxes(img))
