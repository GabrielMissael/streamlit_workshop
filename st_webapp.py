# Importamos las librerias necesarias

import numpy as np # linear algebra
import matplotlib.pyplot as plt # for plotting
from sklearn import datasets, svm, metrics # for machine learning
from sklearn.model_selection import train_test_split # for splitting data
import seaborn as sns # for plotting
from skimage.transform import resize # for resizing images
import streamlit as st
from utils import *

# Estilo de las gráficas
sns.set_theme()

st.markdown('''
    # Detección de dígitos 👋🏽
    ------

    En esta aplicación web, utilizamos un modelo SVM
    (*Support Vector Machine*) para predecir dígitos
    escritos a mano.

    Si quieres probar **tu propio dígito**, puedes hacerlo
    subiendo una imagen que tu hayas generado. Para ello,
    sigue las siguientes instrucciones:
    1. Abre paint
    2. Crea una imagen de 64x64 pixeles
    3. Selecciona un pincel (brush) de 8px de ancho
    4. Dibuja un digito en el centro de la imagen
    5. Guarda la imagen como un archivo .jpg
    6. Sube la imagen en el siguiente apartado

    Al final de la página, podrás subir y aplicar el clasificador
    a tu propia imágen.

    ## Creando y entrenando el modelo ⚙️

    ### Los datos 📊
''')

# Cargamos los datos
digits = datasets.load_digits()
st.write('Tamaño de los datos: ')
st.write(digits.data.shape)

st.markdown('''
En la siguiente figura, podemos ver un ejemplo de cada uno de los
dígitos posibles en el dataset.
''')

fig, ax = plt.subplots(2, 5, figsize = (10, 5))

for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(digits.images[i], cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title('Número: {}'.format(digits.target[i]))

    # Remove ticks from the plot
    ax.set_xticks([])
    ax.set_yticks([])

st.pyplot(fig)

st.markdown('''
Podemos mostrar un ejemplo aleatorio de un dígito en específico.
''')

# Add to columns
col1, col2 = st.columns(2)

with col1:
    digit_select = st.selectbox('Elige un dígito para mostrar un ejemplo aleatroio de este',
        options = [x for x in range(10)])

    if st.button(label = 'Obtener otro ejemplo'):
        with col2:

            # Get indexes of all examples of given digit
            idx = np.where(digits.target == int(digit_select))[0]

            # Pick a random example
            i = np.random.choice(idx)

            img = digits.images[i]
            label = int(digit_select)

            fig, ax = plt.subplots(figsize=(3, 3))
            plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Label: %i' % label)
            plt.xticks([])
            plt.yticks([])

            # Show image
            st.pyplot(fig)

st.markdown('''
### Entrenando el modelo 🦾

Cada modelo puede tener diferentes parámetros. En este caso,
solo eligiremos el parametro $\gamma$ (no te preocupes por su significado)
''')

gamma = st.select_slider(
    'Selecciona el valor de gamma',
    options=[0.1, 0.01, 0.001, 0.0001, 0.00001],
    value = 0.001)

# Aplanamos las imágenes para que sean un vector de 64 elementos
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Creamos el clasificador
clf = svm.SVC(gamma=gamma)

# Separamos los datos en train y test (65% train, 35% test)
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.35, shuffle=False
)

# Entrenamos el clasificador
clf.fit(X_train, y_train)

# Predecimos los valores
predicted = clf.predict(X_test)

st.markdown('''
Dado el valor de $\gamma$ (gamma) elegido, obtenemos las siguientes métricas
(qué tan bien está el performance del modelo)
''')

# Veamos una de las métricas de evaluación
accu = metrics.accuracy_score(y_test, predicted)

# Add to columns
col1, col2 = st.columns(2)

with col1:
    st.metric(label = 'Accuracy', value = accu)

 # Confusion matrix
cm = metrics.confusion_matrix(y_test, predicted)
fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
ax.set_title('Matriz de confusión' + str(gamma))
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor real')

with col2:
    st.pyplot(fig)

#Upload file
st.header('Sube tu imagen 🖼️')
uploaded_file = st.file_uploader("😎", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    img = plt.imread(uploaded_file)

    img_resize = adapt_image(img)

    # Flatten image
    img_resize = img_resize.reshape(1, -1)

    # Predict digit
    pred = clf.predict(img_resize)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(show_resize(img))

    st.markdown('#### Predicción: '+ str(pred[0]))
