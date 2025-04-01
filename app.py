"""
API REST para el clasificador de botones mediante imágenes
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image

# Importar módulos propios
import config
from src.utils.heic_converter import convert_heic_in_directory

app = Flask(__name__)

# Configuración para subida de archivos
UPLOAD_FOLDER = os.path.join(config.BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar a 16MB

# Cargar el modelo al inicio
MODEL = None

def allowed_file(filename):
    """
    Verifica si la extensión del archivo es permitida
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cargar_modelo():
    """
    Carga el modelo entrenado
    """
    global MODEL
    if MODEL is None:
        modelo_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
        print(f"Cargando modelo desde: {modelo_path}")
        MODEL = load_model(modelo_path)
    return MODEL

def predecir_imagen(img_path, modelo):
    """
    Predice la clase de una imagen desde una ruta
    """
    # Convertir de HEIC a JPG si es necesario
    if img_path.lower().endswith(('.heic')):
        print(f"Convirtiendo imagen HEIC: {img_path}")
        ruta_dir = os.path.dirname(img_path) or '.'
        convert_heic_in_directory(ruta_dir, recursive=False)
        img_path = os.path.splitext(img_path)[0] + '.jpg'

    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Hacer predicción
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    
    return clase_predicha, prediccion[0].tolist()

def predecir_imagen_bytes(img_bytes, modelo):
    """
    Predice la clase de una imagen desde bytes
    """
    # Convertir bytes a imagen PIL
    img = Image.open(io.BytesIO(img_bytes))
    
    # Redimensionar
    img = img.resize((config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
    
    # Convertir a array y normalizar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Hacer predicción
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    
    return clase_predicha, prediccion[0].tolist()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar que la API está funcionando
    """
    return jsonify({
        'status': 'ok',
        'modelo': config.MEJOR_MODELO,
        'clases': config.CLASES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones
    """
    # Verificar si hay un archivo en la petición
    if 'file' not in request.files:
        return jsonify({
            'error': 'No se ha enviado ningún archivo'
        }), 400
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return jsonify({
            'error': 'No se ha seleccionado ningún archivo'
        }), 400
    
    # Verificar si el archivo tiene una extensión permitida
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Formato de archivo no soportado. Formatos permitidos: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Cargar el modelo
        modelo = cargar_modelo()
        
        # Dos opciones:
        # 1. Guardar el archivo y procesarlo (mejor para imágenes HEIC)
        if file.filename.lower().endswith('.heic'):
            # Guardar el archivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predecir
            clase_id, probabilidades = predecir_imagen(filepath, modelo)
        # 2. Procesar directamente desde bytes (más rápido para formatos comunes)
        else:
            # Leer bytes del archivo
            img_bytes = file.read()
            
            # Predecir
            clase_id, probabilidades = predecir_imagen_bytes(img_bytes, modelo)
        
        # Preparar respuesta
        return jsonify({
            'clase': config.CLASES[clase_id],
            'clase_id': int(clase_id),
            'probabilidades': {
                config.CLASES[i]: float(prob) 
                for i, prob in enumerate(probabilidades)
            },
            'confianza': float(probabilidades[clase_id])
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la imagen: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Asegurar que el modelo se carga al inicio
    print("Inicializando API...")
    cargar_modelo()
    
    # Iniciar servidor en puerto 8000 (en lugar de 5000 que está ocupado por AirPlay en macOS)
    app.run(host='0.0.0.0', port=8000, debug=False)