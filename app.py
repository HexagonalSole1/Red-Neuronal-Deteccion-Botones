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
import logging

# Importar módulos propios
import config
from src.utils.heic_converter import convert_heic_in_directory
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
CLASS_INDICES = None  # Para almacenar el mapeo de índices a nombres de clase

def allowed_file(filename):
    """
    Verifica si la extensión del archivo es permitida
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verificar_indices_clase():
    """
    Verifica y genera el diccionario de índices de clase
    basado en la estructura de carpetas
    """
    global CLASS_INDICES
    
    # Si ya está definido, reutilizarlo
    if CLASS_INDICES is not None:
        return CLASS_INDICES
    
    # Generar índices de clase a partir de la estructura del directorio de entrenamiento
    train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
    if os.path.exists(train_dir):
        clases = sorted([d for d in os.listdir(train_dir) 
                 if os.path.isdir(os.path.join(train_dir, d))])
        CLASS_INDICES = {clase: i for i, clase in enumerate(clases)}
        logger.info(f"Índices de clase generados: {CLASS_INDICES}")
    else:
        # Si no existe el directorio, usar clases desde config.py
        CLASS_INDICES = {clase: i for i, clase in enumerate(config.CLASES)}
        logger.info(f"Usando índices de clase desde config.py: {CLASS_INDICES}")
    
    return CLASS_INDICES

def cargar_modelo():
    """
    Carga el modelo entrenado
    """
    global MODEL
    if MODEL is None:
        try:
            # Intentar cargar el modelo simple entrenado
            modelo_simple_path = os.path.join(config.MODELS_DIR, 'simple_model_best.h5')
            if os.path.exists(modelo_simple_path):
                logger.info(f"Cargando modelo simple desde: {modelo_simple_path}")
                MODEL = load_model(modelo_simple_path)
                logger.info("Modelo simple cargado correctamente")
            else:
                # Si no existe, intentar con el modelo configurado en config.py
                modelo_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
                logger.info(f"Modelo simple no encontrado. Cargando modelo configurado desde: {modelo_path}")
                MODEL = load_model(modelo_path)
                logger.info("Modelo configurado cargado correctamente")
                
            # Verificar los índices de clase
            verificar_indices_clase()
                
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo: {e}")
    
    return MODEL

def predecir_imagen(img_path, modelo):
    """
    Predice la clase de una imagen desde una ruta
    
    Args:
        img_path: Ruta a la imagen
        modelo: Modelo cargado
        
    Returns:
        clase_id, probabilidades
    """
    try:
        # Convertir de HEIC a JPG si es necesario
        if img_path.lower().endswith(('.heic')):
            logger.info(f"Convirtiendo imagen HEIC: {img_path}")
            ruta_dir = os.path.dirname(img_path) or '.'
            convert_heic_in_directory(ruta_dir, recursive=False)
            img_path = os.path.splitext(img_path)[0] + '.jpg'
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"No se pudo convertir la imagen HEIC a JPG: {img_path}")

        # Cargar y preprocesar la imagen
        img = image.load_img(img_path, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Aplicar el preprocesamiento específico de MobileNetV2
        logger.debug(f"Forma de imagen antes de preprocesamiento: {img_array.shape}")
        img_array_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        logger.debug(f"Forma de imagen después de preprocesamiento: {img_array_processed.shape}")
        
        # Hacer predicción (con output verboso para depuración)
        logger.info("Realizando predicción...")
        prediccion = modelo.predict(img_array_processed, verbose=0)
        logger.debug(f"Forma de predicción: {prediccion.shape}")
        
        clase_predicha = np.argmax(prediccion, axis=1)[0]
        logger.info(f"Clase predicha (ID): {clase_predicha}")
        
        # Registrar probabilidades para depuración
        for i, prob in enumerate(prediccion[0]):
            if prob > 0.05:  # Solo mostrar probabilidades significativas
                clase_nombre = list(CLASS_INDICES.keys())[list(CLASS_INDICES.values()).index(i)]
                logger.debug(f"Clase {clase_nombre} (ID {i}): {prob*100:.2f}%")
        
        return clase_predicha, prediccion[0].tolist()
    
    except Exception as e:
        logger.error(f"Error en predecir_imagen: {e}")
        raise

def predecir_imagen_bytes(img_bytes, modelo):
    """
    Predice la clase de una imagen desde bytes
    
    Args:
        img_bytes: Bytes de la imagen
        modelo: Modelo cargado
        
    Returns:
        clase_id, probabilidades
    """
    try:
        # Convertir bytes a imagen PIL
        img = Image.open(io.BytesIO(img_bytes))
        
        # Redimensionar
        img = img.resize((config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
        
        # Convertir a array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Aplicar el preprocesamiento específico de MobileNetV2
        img_array_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Hacer predicción
        prediccion = modelo.predict(img_array_processed, verbose=0)
        clase_predicha = np.argmax(prediccion, axis=1)[0]
        
        return clase_predicha, prediccion[0].tolist()
    
    except Exception as e:
        logger.error(f"Error en predecir_imagen_bytes: {e}")
        raise

@app.route('/debug-model', methods=['GET'])
def debug_model():
    """
    Endpoint para depurar información del modelo
    """
    try:
        modelo = cargar_modelo()
        return jsonify({
            'model_loaded': modelo is not None,
            'model_type': str(type(modelo)),
            'output_shape': str(modelo.output_shape),
            'class_indices': verificar_indices_clase(),
            'classes_from_config': config.CLASES,
            'input_shape': str(modelo.input_shape)
        })
    except Exception as e:
        return jsonify({
            'error': f'Error al depurar modelo: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar que la API está funcionando
    """
    # Verificar modelo cargado
    modelo_actual = 'simple_model_best.h5' if os.path.exists(os.path.join(config.MODELS_DIR, 'simple_model_best.h5')) else config.MEJOR_MODELO
    
    return jsonify({
        'status': 'ok',
        'modelo': modelo_actual,
        'clases': list(verificar_indices_clase().keys()),
        'version': '2.1'  # Versión actualizada de la API
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones
    """
    # Verificar si hay un archivo en la petición
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No se ha enviado ningún archivo'
        }), 400
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No se ha seleccionado ningún archivo'
        }), 400
    
    # Verificar si el archivo tiene una extensión permitida
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'Formato de archivo no soportado. Formatos permitidos: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Cargar el modelo
        modelo = cargar_modelo()
        clase_indices = verificar_indices_clase()
        
        # Mapeo inverso de índices a nombres de clase
        idx_to_class = {v: k for k, v in clase_indices.items()}
        
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
        
        # Obtener el nombre de la clase
        if clase_id in idx_to_class:
            clase_nombre = idx_to_class[clase_id]
        else:
            logger.error(f"Clase ID {clase_id} no encontrada en el mapeo: {idx_to_class}")
            clase_nombre = f"Desconocido (ID: {clase_id})"
        
        # Preparar respuesta compatible con el frontend
        all_predictions = [
            {
                'class': idx_to_class.get(i, f"Desconocido (ID: {i})"),
                'confidence': float(prob)
            } 
            for i, prob in enumerate(probabilidades)
        ]
        
        # Ordenar predicciones de mayor a menor confianza
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'status': 'ok',
            'prediction': clase_nombre,
            'confidence': float(probabilidades[clase_id]),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        logger.error(f"Error en endpoint /predict: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar la imagen: {str(e)}'
        }), 500
# Endpoint adicional para visualizar información sobre las clases
@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Endpoint para obtener información sobre las clases
    """
    try:
        # Calcular número de ejemplos por clase
        class_counts = {}
        train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
        
        # Usar el mapeo de clase verificado
        class_indices = verificar_indices_clase()
        
        if os.path.exists(train_dir):
            for class_name in class_indices.keys():
                class_dir = os.path.join(train_dir, class_name)
                if os.path.exists(class_dir):
                    class_counts[class_name] = len([f for f in os.listdir(class_dir) 
                                                if os.path.isfile(os.path.join(class_dir, f))])
                else:
                    class_counts[class_name] = 0
        
        return jsonify({
            'classes': list(class_indices.keys()),
            'class_indices': class_indices,
            'total_classes': len(class_indices),
            'class_counts': class_counts,
            'image_size': {
                'height': config.ALTURA_IMAGEN,
                'width': config.ANCHO_IMAGEN,
                'channels': config.CANALES
            }
        })
    except Exception as e:
        logger.error(f"Error en endpoint /classes: {e}")
        return jsonify({
            'error': f'Error al obtener información de clases: {str(e)}'
        }), 500

# Añade esto en las importaciones al inicio del archivo
from flask import render_template, jsonify

# Ruta para ButtonCraft
@app.route('/buttoncraft')
def buttoncraft():
    """Página de ButtonCraft con clasificador integrado"""
    # Información básica para la plantilla (sin depender de módulos de configuración)
    model_info = {
        'num_classes': 6,  # Número fijo de clases de botones
        'classes': ["Botones Metálicos", "Botones de Madera", "Botones Decorativos", 
                   "Botones de Plástico", "Botones para Niños", "Botones Personalizados"],
        'image_size': "224x224"
    }
    
    return render_template('buttoncraft.html', info=model_info)

    return render_template('buttoncraft.html', info=model_info)
if __name__ == '__main__':
    # Configurar nivel de logging para depuración
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Asegurar que el modelo se carga al inicio
    print("=" * 50)
    print("INICIALIZANDO API DE CLASIFICACIÓN DE BOTONES")
    print("=" * 50)
    try:
        modelo = cargar_modelo()
        class_indices = verificar_indices_clase()
        print(f"Modelo cargado correctamente: {type(modelo).__name__}")
        print(f"Número de clases: {len(class_indices)}")
        print(f"Clases disponibles: {list(class_indices.keys())}")
    except Exception as e:
        print(f"Error al inicializar API: {e}")
        print("La API se iniciará, pero las predicciones podrían fallar")
    
    print("\nIniciando servidor en puerto 8000...")
    # Iniciar servidor en puerto 8000 (en lugar de 5000 que está ocupado por AirPlay en macOS)
    app.run(host='0.0.0.0', port=8000, debug=True)  # Debug mode para más información