"""
Configuración global para el proyecto de clasificación de botones
"""

import os
from pathlib import Path

# Rutas principales
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Asegurar que los directorios existen
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Función para detectar clases automáticamente
def detectar_clases():
    """Detecta las clases disponibles en el directorio de entrenamiento"""
    ruta_entrenamiento = os.path.join(DATA_DIR, "entrenamiento")
    
    if not os.path.exists(ruta_entrenamiento):
        print(f"ADVERTENCIA: El directorio {ruta_entrenamiento} no existe.")
        return [], 0
    
    # Obtener subdirectorios (cada uno es una clase)
    clases = [d for d in os.listdir(ruta_entrenamiento) 
              if os.path.isdir(os.path.join(ruta_entrenamiento, d))]
    
    if not clases:
        print(f"ADVERTENCIA: No se encontraron subdirectorios de clases en {ruta_entrenamiento}.")
        return [], 0
    
    print(f"Clases detectadas: {clases}")
    return clases, len(clases)

# Detectar clases automáticamente
CLASES, NUM_CLASES = detectar_clases()

# Parámetros de las imágenes
ALTURA_IMAGEN = 150
ANCHO_IMAGEN = 150
CANALES = 3

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCAS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# En config.py
DATA_AUGMENTATION = {
    'rotation_range': 40,  # Aumentar rotación
    'width_shift_range': 0.3,  # Más desplazamiento
    'height_shift_range': 0.3,
    'shear_range': 0.3,
    'zoom_range': 0.3,
    'horizontal_flip': True,
    'vertical_flip': True,  # Añadir flip vertical
    'brightness_range': [0.7, 1.3],  # Variar brillo
    'fill_mode': 'nearest'
}

# Nombres de archivos para los modelos guardados
MEJOR_MODELO = "mejor_modelo_transfer.h5"
MODELO_FINAL = "modelo_final_botones.h5"