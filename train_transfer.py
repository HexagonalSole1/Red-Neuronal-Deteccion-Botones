"""
Script para entrenar el modelo de clasificación de botones usando transfer learning
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Importar módulos propios
import config
from data.preprocesar_datos import crear_generadores_datos
from models.model_transfer import crear_modelo_transfer, crear_modelo_transfer_fine_tuning
from src.utils.visualization import plot_training_history
from src.utils.heic_converter import prepare_image_directories

# Configuración para reproducibilidad
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def train_transfer_model(fine_tuning=False, usar_pesos_clase=True):
    """
    Entrena el modelo de clasificación de botones usando transfer learning
    
    Args:
        fine_tuning: Si True, usa fine-tuning en las últimas capas del modelo base
        usar_pesos_clase: Si True, aplica pesos de clase para equilibrar el entrenamiento
    """
    print("=" * 50)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE BOTONES CON TRANSFER LEARNING")
    print("=" * 50)
    
    # Configurar semilla para reproducibilidad
    set_seed()
    
    # Crear directorios si no existen
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Convertir imágenes HEIC a JPG si existen
    print("\n[1/5] Preparando imágenes (convirtiendo HEIC si existen)...")
    prepare_image_directories(config)
    
    # Cargar los datos - necesitamos normalizar los datos según el modelo preentrenado
    # Los generadores de datos se configurarán para usar la normalización correcta
    print("\n[2/5] Cargando y preparando datos...")
    train_generator, validation_generator, test_generator = crear_generadores_datos(
        config, 
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    
    # Crear el modelo
    print("\n[3/5] Creando modelo con transfer learning...")
    if fine_tuning:
        print("Usando fine-tuning en las últimas capas del modelo base")
        model = crear_modelo_transfer_fine_tuning(config)
        modelo_nombre = "transfer_fine_tuned"
    else:
        print("Usando modelo base congelado")
        model = crear_modelo_transfer(config)
        modelo_nombre = "transfer"
    
    model.summary()
    
    # Callback para guardar el mejor modelo
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.MODELS_DIR, f"mejor_modelo_{modelo_nombre}.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping para evitar overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,  # Más paciencia para transfer learning
        restore_best_weights=True
    )
    
    # Reducir learning rate cuando se estanca el entrenamiento
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Calcular pesos de clase si es necesario
    class_weights = None
    if usar_pesos_clase:
        # Probar con un balance más equilibrado
        class_weights = {0: 1.2, 1: 1.0}  # Dar un ligero peso adicional a botones azules
        print("\nUsando pesos de clase para equilibrar el entrenamiento:")
        for clase, peso in class_weights.items():
            nombre_clase = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(clase)]
            print(f"  - Clase '{nombre_clase}' (índice {clase}): peso {peso}")
    
    # Entrenamiento
    print("\n[4/5] Entrenando modelo...")
    start_time = datetime.now()
    
    # Usar más épocas para transfer learning
    epocas = config.EPOCAS * 2 if fine_tuning else config.EPOCAS
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        epochs=epocas,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // config.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )
    training_time = datetime.now() - start_time
    
    # Guardar el modelo final
    model.save(os.path.join(config.MODELS_DIR, f"modelo_final_{modelo_nombre}.h5"))
    
    # Actualizar el archivo config.py para usar el nuevo mejor modelo por defecto
    with open('config.py', 'r') as file:
        config_content = file.read()
    
    # Reemplazar la línea MEJOR_MODELO
    config_content = config_content.replace(
        f'MEJOR_MODELO = "{config.MEJOR_MODELO}"', 
        f'MEJOR_MODELO = "mejor_modelo_{modelo_nombre}.h5"'
    )
    
    # Guardar el archivo config actualizado
    with open('config.py', 'w') as file:
        file.write(config_content)
    
    # Visualizar historial de entrenamiento
    print("\n[5/5] Generando visualizaciones...")
    plot_training_history(history, config.OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print(f"ENTRENAMIENTO COMPLETADO EN {training_time}")
    print("=" * 50)
    print(f"Mejor modelo guardado en: {os.path.join(config.MODELS_DIR, f'mejor_modelo_{modelo_nombre}.h5')}")
    print(f"Modelo final guardado en: {os.path.join(config.MODELS_DIR, f'modelo_final_{modelo_nombre}.h5')}")
    print(f"Visualizaciones guardadas en: {config.OUTPUT_DIR}")
    print(f"Archivo config.py actualizado para usar el nuevo modelo por defecto")
    
    return model, history, train_generator, validation_generator, test_generator

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar clasificador de botones con transfer learning')
    parser.add_argument('--fine-tuning', action='store_true', 
                        help='Usar fine-tuning en las últimas capas del modelo base')
    parser.add_argument('--no-pesos', action='store_true',
                        help='No usar pesos de clase en el entrenamiento')
    
    args = parser.parse_args()
    
    # Entrenar modelo con las opciones especificadas
    train_transfer_model(fine_tuning=args.fine_tuning, usar_pesos_clase=not args.no_pesos)