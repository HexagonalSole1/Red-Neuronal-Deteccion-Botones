"""
Script para visualización y documentación del dataset
Este script genera visualizaciones del dataset y documenta las clases
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import shutil
from collections import Counter

# Importar configuración
import config
from src.utils.heic_converter import prepare_image_directories

def visualize_dataset():
    """
    Analiza y visualiza el dataset de botones
    Genera documentación visual para el informe
    """
    print("=" * 50)
    print("VISUALIZACIÓN Y DOCUMENTACIÓN DEL DATASET")
    print("=" * 50)
    
    # Asegurar directorios de salida
    output_dir = os.path.join(config.OUTPUT_DIR, "dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir imágenes HEIC si existen
    prepare_image_directories(config)
    
    # Directorios
    train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
    test_dir = os.path.join(config.DATA_DIR, 'prueba')
    
    # Verificar que los directorios existen
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"El directorio de entrenamiento no existe: {train_dir}")
    
    # 1. Detectar clases y contar ejemplos
    class_stats = {}
    total_images = 0
    
    for class_name in config.CLASES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            # Contar imágenes
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = [f for f in os.listdir(class_dir) 
                          if os.path.isfile(os.path.join(class_dir, f)) and 
                          os.path.splitext(f.lower())[1] in valid_extensions]
            
            num_images = len(image_files)
            total_images += num_images
            
            # Obtener información de dimensiones y colores muestreando algunas imágenes
            sample_size = min(10, num_images)
            samples = random.sample(image_files, sample_size) if num_images > 0 else []
            
            dimensions = []
            color_ranges = []
            
            for img_file in samples:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path)
                img_array = np.array(img)
                
                dimensions.append(img.size)
                
                # Análisis de rango de colores
                if len(img_array.shape) == 3:  # Si es RGB
                    mins = np.min(img_array, axis=(0, 1))
                    maxs = np.max(img_array, axis=(0, 1))
                    means = np.mean(img_array, axis=(0, 1))
                    stds = np.std(img_array, axis=(0, 1))
                    
                    color_ranges.append({
                        'min': mins,
                        'max': maxs,
                        'mean': means,
                        'std': stds
                    })
            
            # Guardar estadísticas
            class_stats[class_name] = {
                'count': num_images,
                'percentage': (num_images / total_images * 100) if total_images > 0 else 0,
                'dimensions': dimensions,
                'color_ranges': color_ranges,
                'sample_files': samples
            }
    
    # 2. Generar visualizaciones
    
    # 2.1 Distribución de clases
    plt.figure(figsize=(10, 6))
    counts = [stats['count'] for _, stats in class_stats.items()]
    plt.bar(class_stats.keys(), counts)
    plt.title('Distribución de clases en el dataset')
    plt.xlabel('Clase')
    plt.ylabel('Número de imágenes')
    plt.xticks(rotation=45)
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribucion_clases.png'))
    
    # 2.2 Ejemplos de cada clase
    for class_name, stats in class_stats.items():
        # Tomar hasta 3 ejemplos de cada clase
        num_examples = min(3, len(stats['sample_files']))
        
        if num_examples > 0:
            plt.figure(figsize=(15, 5))
            
            for i in range(num_examples):
                img_file = stats['sample_files'][i]
                img_path = os.path.join(train_dir, class_name, img_file)
                
                plt.subplot(1, num_examples, i+1)
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.title(f'{class_name} - Ejemplo {i+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ejemplos_{class_name}.png'))
    
    # 3. Generar reporte de estadísticas
    report = "# Análisis del Dataset\n\n"
    report += f"Total de imágenes: {total_images}\n\n"
    
    for class_name, stats in class_stats.items():
        report += f"## Clase: {class_name}\n\n"
        report += f"* Número de imágenes: {stats['count']}\n"
        report += f"* Porcentaje del dataset: {stats['percentage']:.2f}%\n"
        
        if stats['dimensions']:
            # Encontrar dimensiones más comunes
            dims = Counter(stats['dimensions'])
            most_common = dims.most_common(1)[0]
            report += f"* Dimensiones más comunes: {most_common[0][0]}x{most_common[0][1]} (frecuencia: {most_common[1]})\n"
        
        if stats['color_ranges']:
            # Promedio de medias y desviaciones estándar
            mean_of_means = np.mean([cr['mean'] for cr in stats['color_ranges']], axis=0)
            mean_of_stds = np.mean([cr['std'] for cr in stats['color_ranges']], axis=0)
            
            report += f"* Promedio de valores RGB: R={mean_of_means[0]:.1f}, G={mean_of_means[1]:.1f}, B={mean_of_means[2]:.1f}\n"
            report += f"* Desviación estándar RGB: R={mean_of_stds[0]:.1f}, G={mean_of_stds[1]:.1f}, B={mean_of_stds[2]:.1f}\n"
        
        report += "\n"
    
    # Guardar reporte
    with open(os.path.join(output_dir, 'dataset_analysis.md'), 'w') as f:
        f.write(report)
    
    print(f"\nAnálisis del dataset completado. Resultados guardados en: {output_dir}")
    
    return class_stats


def prepare_sample_dataset(force_recreate=False):
    """
    Prepara un conjunto pequeño de datos para pruebas y demos
    Extrae muestras balanceadas de cada clase
    
    Args:
        force_recreate: Si se debe forzar la recreación del dataset de muestra
    """
    samples_dir = os.path.join(config.DATA_DIR, 'muestras')
    
    # Verificar si ya existe
    if os.path.exists(samples_dir) and not force_recreate:
        print(f"El directorio de muestras ya existe: {samples_dir}")
        return samples_dir
    
    # Crear directorio de muestras
    os.makedirs(samples_dir, exist_ok=True)
    
    # Obtener directorios de entrenamiento
    train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
    
    # Número de muestras por clase
    num_samples = 5
    
    for class_name in config.CLASES:
        # Crear directorio para esta clase
        class_dir = os.path.join(samples_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Directorio de origen
        source_dir = os.path.join(train_dir, class_name)
        
        if os.path.exists(source_dir):
            # Obtener lista de archivos
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = [f for f in os.listdir(source_dir) 
                          if os.path.isfile(os.path.join(source_dir, f)) and 
                          os.path.splitext(f.lower())[1] in valid_extensions]
            
            # Seleccionar muestras aleatorias
            selected_samples = random.sample(image_files, min(num_samples, len(image_files)))
            
            # Copiar archivos
            for sample in selected_samples:
                shutil.copy2(
                    os.path.join(source_dir, sample),
                    os.path.join(class_dir, sample)
                )
    
    print(f"Dataset de muestras creado en: {samples_dir}")
    return samples_dir


def main():
    # Visualizar dataset y generar estadísticas
    class_stats = visualize_dataset()
    
    # Preparar conjunto de muestras para demo
    samples_dir = prepare_sample_dataset()
    
    print("\nResumen del dataset:")
    for class_name, stats in class_stats.items():
        print(f"- {class_name}: {stats['count']} imágenes ({stats['percentage']:.2f}%)")


if __name__ == "__main__":
    main()