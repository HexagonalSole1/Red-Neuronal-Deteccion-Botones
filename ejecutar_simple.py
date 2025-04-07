"""
Versión simplificada del script ejecutar.py que no depende de OpenCV
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib
from datetime import datetime

# Importar módulos propios
import config
from k_fold_validation import train_with_kfold_validation

def create_project_structure():
    """Crea la estructura de directorios del proyecto si no existe"""
    directories = [
        config.DATA_DIR,
        os.path.join(config.DATA_DIR, 'entrenamiento'),
        os.path.join(config.DATA_DIR, 'prueba'),
        config.MODELS_DIR,
        config.OUTPUT_DIR,
        os.path.join(config.OUTPUT_DIR, 'kfold')
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
    print(f"Matplotlib: {matplotlib.__version__}")
    
    # Información sobre GPU
    if tf.config.list_physical_devices('GPU'):
        print("\nGPU disponible para entrenamiento:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
    else:
        print("\nNo se detectaron GPUs. El entrenamiento se realizará en CPU.")
    
    print("\n")

def run_simplified_pipeline(model_type='transfer', k_folds=5, fine_tuning=False, epochs=None, batch_size=None):
    """
    Ejecuta un pipeline simplificado: solo entrenamiento con validación cruzada
    
    Args:
        model_type: Tipo de modelo a entrenar ('basic', 'cnn', o 'transfer')
        k_folds: Número de folds para validación cruzada
        fine_tuning: Si se debe aplicar fine-tuning al modelo de transfer learning
        epochs: Número de épocas para entrenamiento
        batch_size: Tamaño de batch para entrenamiento
    """
    start_time = datetime.now()
    
    print("=" * 50)
    print("CLASIFICADOR DE BOTONES - PIPELINE SIMPLIFICADO")
    print("=" * 50)
    print(f"Configuración: modelo={model_type}, k_folds={k_folds}, fine_tuning={fine_tuning}")
    print("=" * 50)
    
    # 1. Crear estructura de proyecto
    print("\n[1/2] Creando estructura de proyecto...")
    create_project_structure()
    
    print("\n[2/2] Iniciando entrenamiento con validación cruzada...")
    results = train_with_kfold_validation(
        k_folds=k_folds,
        model_type=model_type,
        fine_tuning=fine_tuning,
        epochs=epochs,
        batch_size=batch_size
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
    parser = argparse.ArgumentParser(description='Clasificador de Botones - Pipeline simplificado')
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
    
    # Ejecutar pipeline simplificado
    run_simplified_pipeline(
        model_type=args.model,
        k_folds=args.folds,
        fine_tuning=args.fine_tuning,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()