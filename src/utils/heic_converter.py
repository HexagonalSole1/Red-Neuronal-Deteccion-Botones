"""
Módulo para convertir imágenes HEIC a JPG de forma automática
"""

import os
import glob
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    logger.warning("pillow-heif no está instalado. La conversión HEIC->JPG no estará disponible.")
    logger.warning("Instala con: pip install pillow-heif")

def convert_heic_in_directory(directory, recursive=True):
    """
    Convierte todas las imágenes HEIC en un directorio a formato JPG
    
    Args:
        directory: Directorio donde buscar y convertir imágenes HEIC
        recursive: Si True, busca imágenes en subdirectorios
    
    Returns:
        num_converted: Número de imágenes convertidas
    """
    if not HEIF_SUPPORT:
        logger.warning("No se pueden convertir imágenes HEIC. Instala pillow-heif primero.")
        return 0
    
    # Buscar archivos HEIC
    heic_files = []
    if recursive:
        for ext in ['*.heic', '*.HEIC']:
            pattern = os.path.join(directory, '**', ext)
            heic_files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in ['*.heic', '*.HEIC']:
            pattern = os.path.join(directory, ext)
            heic_files.extend(glob.glob(pattern))
    
    if not heic_files:
        logger.info(f"No se encontraron imágenes HEIC en {directory}")
        return 0
    
    logger.info(f"Encontradas {len(heic_files)} imágenes HEIC para convertir.")
    
    # Convertir cada archivo HEIC
    num_converted = 0
    for heic_file in heic_files:
        try:
            # Construir el nombre del archivo de salida
            jpg_path = os.path.splitext(heic_file)[0] + '.jpg'
            
            # Verificar si el JPG ya existe y es más reciente que el HEIC
            if os.path.exists(jpg_path) and os.path.getmtime(jpg_path) > os.path.getmtime(heic_file):
                logger.debug(f"Saltando {heic_file} - JPG ya existe y es más reciente")
                continue
                
            # Abrir imagen HEIC y guardar como JPG
            img = Image.open(heic_file)
            img = img.convert('RGB')  # Asegurar formato RGB
            img.save(jpg_path, 'JPEG', quality=95)
            
            num_converted += 1
            logger.info(f"Convertido: {heic_file} -> {jpg_path}")
        except Exception as e:
            logger.error(f"Error al convertir {heic_file}: {e}")
    
    logger.info(f"Conversión completada. {num_converted} imágenes convertidas.")
    return num_converted

def prepare_image_directories(config):
    """
    Prepara los directorios de imágenes convirtiendo archivos HEIC en ellos
    
    Args:
        config: Módulo de configuración del proyecto
        
    Returns:
        total_converted: Número total de imágenes convertidas
    """
    # Directorios a procesar
    dirs_to_process = [
        os.path.join(config.DATA_DIR, "entrenamiento"),
        os.path.join(config.DATA_DIR, "prueba")
    ]
    
    total_converted = 0
    for directory in dirs_to_process:
        if os.path.exists(directory):
            logger.info(f"Procesando directorio: {directory}")
            total_converted += convert_heic_in_directory(directory, recursive=True)
        else:
            logger.warning(f"El directorio {directory} no existe.")
    
    return total_converted