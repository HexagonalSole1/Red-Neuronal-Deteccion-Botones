"""
Script principal para ejecutar el entrenamiento y evaluación del clasificador de botones
Este script integra todos los componentes del proyecto según los requisitos de la asignatura
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import matplotlib


# Importar módulos propios
import config
from dataset_visualization import visualize_dataset, prepare_sample_dataset
from k_fold_validation import train_with_kfold_validation
from src.utils.results_visualization import (
    plot_kfold_results, 
    plot_confusion_matrix, 
    visualize_predictions, 
    apply_gradcam,
    create_results_collage
)

def create_project_structure():
    """Crea la estructura de directorios del proyecto si no existe"""
    directories = [
        config.DATA_DIR,
        os.path.join(config.DATA_DIR, 'entrenamiento'),
        os.path.join(config.DATA_DIR, 'prueba'),
        config.MODELS_DIR,
        config.OUTPUT_DIR,
        os.path.join(config.OUTPUT_DIR, 'dataset'),
        os.path.join(config.OUTPUT_DIR, 'kfold'),
        os.path.join(config.OUTPUT_DIR, 'visualizaciones')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directorio creado/verificado: {directory}")

def show_system_info():
    """Muestra información sobre el sistema y las versiones instaladas"""
    print("=" * 50)
    print("INFORMACIÓN DEL SISTEMA")
    print("=" * 50)
    
    print(f"Python: {os.sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    import matplotlib
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Pandas: {pd.__version__}")
    
    # Información sobre GPU
    if tf.config.list_physical_devices('GPU'):
        print("\nGPU disponible para entrenamiento:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
        
        # Información de CUDA
        if hasattr(tf.sysconfig, 'get_build_info'):
            build_info = tf.sysconfig.get_build_info()
            if 'cuda_version' in build_info:
                print(f"CUDA: {build_info['cuda_version']}")
            if 'cudnn_version' in build_info:
                print(f"cuDNN: {build_info['cudnn_version']}")
    else:
        print("\nNo se detectaron GPUs. El entrenamiento se realizará en CPU.")
    
    print("\n")

def run_full_pipeline(model_type='transfer', k_folds=5, fine_tuning=False, epochs=None, batch_size=None):
    """
    Ejecuta el pipeline completo: visualización de datos, entrenamiento con validación cruzada,
    evaluación y visualización de resultados
    
    Args:
        model_type: Tipo de modelo a entrenar ('basic', 'cnn', o 'transfer')
        k_folds: Número de folds para validación cruzada
        fine_tuning: Si se debe aplicar fine-tuning al modelo de transfer learning
        epochs: Número de épocas para entrenamiento
        batch_size: Tamaño de batch para entrenamiento
    """
    start_time = datetime.now()
    
    print("=" * 50)
    print("CLASIFICADOR DE BOTONES - PIPELINE COMPLETO")
    print("=" * 50)
    print(f"Configuración: modelo={model_type}, k_folds={k_folds}, fine_tuning={fine_tuning}")
    print("=" * 50)
    
    # 1. Crear estructura de proyecto
    print("\n[1/5] Creando estructura de proyecto...")
    create_project_structure()
    
    # 2. Visualizar dataset
    print("\n[2/5] Analizando y visualizando dataset...")
    class_stats = visualize_dataset()
    
    # 3. Entrenamiento con validación cruzada
    print("\n[3/5] Iniciando entrenamiento con validación cruzada...")
    results = train_with_kfold_validation(
        k_folds=k_folds,
        model_type=model_type,
        fine_tuning=fine_tuning,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 4. Visualización de resultados
    print("\n[4/5] Generando visualizaciones de resultados...")
    output_dir = os.path.join(config.OUTPUT_DIR, 'kfold')
    plot_kfold_results(results['fold_metrics'], output_dir)
    
    # 5. Demostración con muestras
    print("\n[5/5] Preparando demostración con muestras...")
    samples_dir = prepare_sample_dataset()
    
    # Cargar modelo entrenado
    model = results['best_model']
    
    # Generar visualizaciones con muestras
    visualization_dir = os.path.join(config.OUTPUT_DIR, 'visualizaciones')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Recolectar imágenes de muestra para el collage
    sample_images = []
    for class_name in config.CLASES:
        class_dir = os.path.join(samples_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                     if os.path.isfile(os.path.join(class_dir, f)) and 
                     f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Tomar hasta 2 imágenes de cada clase
            if images:
                selected = np.random.choice(images, size=min(2, len(images)), replace=False)
                sample_images.extend(selected)
    
    # Crear collage de resultados
    if sample_images:
        create_results_collage(
            model, 
            sample_images, 
            os.path.join(visualization_dir, 'demo_results.png')
        )
    
    # Tiempo total
    total_time = datetime.now() - start_time
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETADO")
    print("=" * 50)
    print(f"Tiempo total: {total_time}")
    print(f"Precisión promedio: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Mejor modelo guardado en: {results['best_model_path']}")
    print("=" * 50)

def main():
    # Mostrar información del sistema
    show_system_info()
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Clasificador de Botones - Pipeline completo')
    parser.add_argument('--model', type=str, default='transfer', 
                        choices=['basic', 'cnn', 'transfer'],
                        help='Tipo de modelo a entrenar (default: transfer)')
    parser.add_argument('--folds', type=int, default=5, 
                        help='Número de folds para validación cruzada (default: 5)')
    parser.add_argument('--fine-tuning', action='store_true', 
                        help='Usar fine-tuning para modelo de transfer learning')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas para entrenamiento')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Tamaño de batch para entrenamiento')
    
    args = parser.parse_args()
    
    # Ejecutar pipeline completo
    run_full_pipeline(
        model_type=args.model,
        k_folds=args.folds,
        fine_tuning=args.fine_tuning,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()