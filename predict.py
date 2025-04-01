"""
Script para hacer predicciones con el modelo entrenado
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Importar módulos propios
import config
from src.utils.heic_converter import convert_heic_in_directory

def cargar_modelo(modelo_path=None):
    """
    Carga un modelo entrenado
    
    Args:
        modelo_path: Ruta al modelo. Si es None, se usa el mejor modelo guardado.
        
    Returns:
        modelo cargado
    """
    if modelo_path is None:
        modelo_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
    
    print(f"Cargando modelo desde: {modelo_path}")
    return load_model(modelo_path)

def predecir_imagen(ruta_imagen, modelo):
    """
    Predice la clase de una imagen
    
    Args:
        ruta_imagen: Ruta a la imagen
        modelo: Modelo cargado
        
    Returns:
        clase_id, probabilidades
    """
    # Convertir de HEIC a JPG si es necesario
    if ruta_imagen.lower().endswith(('.heic')):
        print(f"Convirtiendo imagen HEIC: {ruta_imagen}")
        ruta_dir = os.path.dirname(ruta_imagen) or '.'
        convert_heic_in_directory(ruta_dir, recursive=False)
        ruta_imagen = os.path.splitext(ruta_imagen)[0] + '.jpg'
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"No se pudo convertir la imagen HEIC a JPG: {ruta_imagen}")
    
    # Cargar y preprocesar la imagen
    img = image.load_img(ruta_imagen, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Hacer predicción
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    
    return clase_predicha, prediccion[0]

def mostrar_prediccion(ruta_imagen, clase_id, probabilidades, guardar=False):
    """
    Muestra la imagen con la predicción
    
    Args:
        ruta_imagen: Ruta a la imagen
        clase_id: ID de la clase predicha
        probabilidades: Probabilidades para cada clase
        guardar: Si se debe guardar la imagen
    """
    # Convertir de HEIC a JPG si es necesario
    if ruta_imagen.lower().endswith(('.heic')):
        ruta_dir = os.path.dirname(ruta_imagen) or '.'
        convert_heic_in_directory(ruta_dir, recursive=False)
        ruta_imagen = os.path.splitext(ruta_imagen)[0] + '.jpg'
    
    # Cargar imagen
    img = image.load_img(ruta_imagen, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    
    # Mostrar imagen
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicción: {config.CLASES[clase_id]}")
    plt.axis('off')
    
    # Mostrar probabilidades
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(config.CLASES))
    plt.barh(y_pos, probabilidades * 100)
    plt.yticks(y_pos, config.CLASES)
    plt.xlabel('Probabilidad (%)')
    plt.title('Probabilidades por clase')
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if guardar:
        img_name = os.path.basename(ruta_imagen).split('.')[0]
        plt.savefig(os.path.join(config.OUTPUT_DIR, f'prediccion_{img_name}.png'))
        plt.close()
    else:
        plt.show()

def main():
    """
    Función principal
    """
    print("=" * 50)
    print("PREDICCIÓN CON CLASIFICADOR DE BOTONES")
    print("=" * 50)
    
    # Cargar modelo
    modelo = cargar_modelo()
    
    # Obtener ruta a la imagen
    ruta_imagen = input("\nIngresa la ruta a la imagen del botón: ")
    
    # Verificar que la imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"Error: La imagen no existe en la ruta especificada: {ruta_imagen}")
        return
    
    # Hacer predicción
    print("\nRealizando predicción...")
    clase_id, probabilidades = predecir_imagen(ruta_imagen, modelo)
    
    # Mostrar resultados
    print("\nResultados de la predicción:")
    print(f"Clase predicha: {config.CLASES[clase_id]}")
    print("\nProbabilidades por clase:")
    for i, clase in enumerate(config.CLASES):
        print(f"{clase}: {probabilidades[i]*100:.2f}%")
    
    # Mostrar la imagen con la predicción
    print("\nMostrando visualización...")
    mostrar_prediccion(ruta_imagen, clase_id, probabilidades)
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()