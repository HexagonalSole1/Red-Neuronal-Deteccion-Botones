"""
Módulo que define la arquitectura del modelo de transfer learning
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def crear_modelo_transfer(config):
    """
    Crea un modelo de clasificación de botones usando transfer learning con MobileNetV2
    Usa el modelo preentrenado con pesos de ImageNet y congela todas las capas base
    
    Args:
        config: Módulo de configuración
        
    Returns:
        Modelo compilado
    """
    # Crear modelo base con pesos pre-entrenados de ImageNet
    base_model = MobileNetV2(
        input_shape=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN, config.CANALES),
        include_top=False,  # No incluir la capa de clasificación final
        weights='imagenet'  # Usar pesos pre-entrenados
    )
    
    # Congelar todas las capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Añadir capas personalizadas para la clasificación de botones
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida con softmax para clasificación multiclase
    predictions = Dense(config.NUM_CLASES, activation='softmax')(x)
    
    # Crear modelo
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def crear_modelo_transfer_fine_tuning(config):
    """
    Crea un modelo de clasificación de botones usando transfer learning con MobileNetV2
    Usa el modelo preentrenado con pesos de ImageNet y permite fine-tuning en las últimas capas
    
    Args:
        config: Módulo de configuración
        
    Returns:
        Modelo compilado
    """
    # Crear modelo base con pesos pre-entrenados de ImageNet
    base_model = MobileNetV2(
        input_shape=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN, config.CANALES),
        include_top=False,  # No incluir la capa de clasificación final
        weights='imagenet'  # Usar pesos pre-entrenados
    )
    
    # Congelar todas las capas excepto las últimas del modelo base
    # Esto permite hacer fine-tuning en las capas superiores del modelo base
    fine_tune_at = len(base_model.layers) - 30  # Descongelar las últimas 30 capas
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True
    
    # Añadir capas personalizadas para la clasificación de botones
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida con softmax para clasificación multiclase
    predictions = Dense(config.NUM_CLASES, activation='softmax')(x)
    
    # Crear modelo
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilar modelo con un learning rate más bajo para fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Imprimir resumen de capas entrenables
    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    non_trainable_count = sum(1 for layer in model.layers if not layer.trainable)
    
    print(f"Capas totales en el modelo: {len(model.layers)}")
    print(f"Capas entrenables: {trainable_count}")
    print(f"Capas no entrenables: {non_trainable_count}")
    
    return model