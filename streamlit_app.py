import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
import wget
import zipfile
import os
import contextlib

st.write(f"TensorFlow version: {tf.__version__}")
# Descargar y descomprimir el modelo si no existe
def download_and_extract_model():
    model_url = 'https://dl.dropboxusercontent.com/s/2273lf2jq335jp4qzwy5h/Pediatric-Pneumonia-Chest-Xray1.zip?rlkey=57cn2gp1am6ixc4ztfs7rc4hy&st=nwj7kinr'
    zip_path = 'Pediatric-Pneumonia-Chest-Xray1'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return False

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    return os.path.join(extract_folder, 'best_model.keras')

modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if not modelo_path or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo")
else:
    st.success("Archivo del modelo encontrado")

# Definir el modelo base InceptionResNetV2
baseModel = InceptionResNetV2(weights=None, include_top=False, input_shape=(299, 299, 3))
baseModel.trainable = False

# Añadir capas de clasificación
x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=baseModel.input, outputs=predictions)

# Cargar los pesos del modelo desde el archivo .keras
try:
    model.load_weights(modelo_path, skip_mismatch=True)
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción con redirección de salida para evitar UnicodeEncodeError
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('El modelo predice que tiene **Neumonía**.')
    else:
        st.success('El modelo predice que **No** tiene **Neumonía**.')
