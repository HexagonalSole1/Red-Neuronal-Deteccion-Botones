"""
Módulo que define las arquitecturas de modelos CNN propios
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def crear_modelo_cnn_basico(config):
    """
    Crea un modelo básico de red neuronal convolucional para clasificación de botones
    
    Args:
        config: Módulo de configuración
        
    Returns:
        Modelo compilado
    """
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN, config.CANALES)),
        MaxPooling2D((2, 2)),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Aplanar para pasar a capas densas
        Flatten(),
        
        # Capas densas
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(config.NUM_CLASES, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def crear_modelo_cnn(config):
    """
    Crea un modelo mejorado de red neuronal convolucional para clasificación de botones
    
    Args:
        config: Módulo de configuración
        
    Returns:
        Modelo compilado
    """
    model = Sequential([
        # Primera capa convolucional con normalización
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN, config.CANALES)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Cuarta capa convolucional
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Aplanar para pasar a capas densas
        Flatten(),
        
        # Capas densas
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(config.NUM_CLASES, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def crear_modelo_cnn_avanzado(config):
    """
    Crea un modelo avanzado de red neuronal convolucional para clasificación de botones
    Usa bloques residuales simples para mejorar el flujo de gradientes
    
    Args:
        config: Módulo de configuración
        
    Returns:
        Modelo compilado
    """
    def residual_block(x, filters, kernel_size=3, stride=1, use_bias=True, scale=True):
        """Bloque residual básico"""
        residual = x
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        
        if stride != 1 or residual.shape[-1] != filters:
            residual = Conv2D(filters, 1, strides=stride, padding='same', use_bias=use_bias)(residual)
            residual = BatchNormalization()(residual)
        
        if scale:
            x = tf.keras.layers.Lambda(lambda inputs: inputs[0] + inputs[1])([x, residual])
        else:
            x = tf.keras.layers.Add()([x, residual])
        
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    # Definir entrada
    inputs = tf.keras.layers.Input(shape=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN, config.CANALES))
    
    # Capa inicial
    x = Conv2D(64, 7, strides=2, padding='same', use_bias=True)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Bloques residuales
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    # Global pooling y clasificación
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(config.NUM_CLASES, activation='softmax')(x)
    
    # Crear modelo
    model = tf.keras.models.Model(inputs, outputs)
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model