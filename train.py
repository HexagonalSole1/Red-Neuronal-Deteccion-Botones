"""
Script principal para entrenar el modelo de clasificación de botones
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Importar módulos propios
import config
from data.preprocesar_datos import crear_generadores_datos
from models.modelo_cnn import crear_modelo_cnn
from src.utils.visualization import plot_training_history

# Configuración para reproducibilidad
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def train_model():
    """
    Entrena el modelo de clasificación de botones
    """
    print("=" * 50)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE BOTONES")
    print("=" * 50)
    
    # Configurar semilla para reproducibilidad
    set_seed()
    
    # Crear directorios si no existen
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Cargar los datos
    print("\n[1/4] Cargando y preparando datos...")
    train_generator, validation_generator, test_generator = crear_generadores_datos(config)
    
    # Crear el modelo
    print("\n[2/4] Creando modelo CNN...")
    model = crear_modelo_cnn(config)
    model.summary()
    
    # Callback para guardar el mejor modelo
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.MODELS_DIR, config.MEJOR_MODELO),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping para evitar overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Reducir learning rate cuando se estanca el entrenamiento
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Entrenamiento
    print("\n[3/4] Entrenando modelo...")
    start_time = datetime.now()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        epochs=config.EPOCAS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // config.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    training_time = datetime.now() - start_time
    
    # Guardar el modelo final
    model.save(os.path.join(config.MODELS_DIR, config.MODELO_FINAL))
    
    # Visualizar historial de entrenamiento
    print("\n[4/4] Generando visualizaciones...")
    plot_training_history(history, config.OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print(f"ENTRENAMIENTO COMPLETADO EN {training_time}")
    print("=" * 50)
    print(f"Mejor modelo guardado en: {os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)}")
    print(f"Modelo final guardado en: {os.path.join(config.MODELS_DIR, config.MODELO_FINAL)}")
    print(f"Visualizaciones guardadas en: {config.OUTPUT_DIR}")
    
    return model, history, train_generator, validation_generator, test_generator

if __name__ == "__main__":
    train_model()