"""
Script simplificado para entrenar un clasificador de botones
Este script entrena un modelo desde cero y lo guarda en un formato compatible
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importar configuración
import config

def train_simple_model():
    """
    Entrena un modelo simple de clasificación de botones
    """
    print("=" * 50)
    print("ENTRENAMIENTO SIMPLE DEL CLASIFICADOR DE BOTONES")
    print("=" * 50)
    
    # Crear directorios si no existen
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Parámetros
    img_height, img_width = config.ALTURA_IMAGEN, config.ANCHO_IMAGEN
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCAS
    
    # Directorios
    train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
    test_dir = os.path.join(config.DATA_DIR, 'prueba')
    
    # Verificar directorios
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"El directorio de entrenamiento no existe: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"El directorio de prueba no existe: {test_dir}")
    
    print("\n[1/5] Configurando generadores de datos...")
    
    # Preprocesamiento para MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Generador para entrenamiento con augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Generador para prueba
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Crear generadores
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Importante para mantener orden para matriz de confusión
    )
    
    # Obtener número de clases
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    # Guardar mapeo de índices a clases para futuras referencias
    print(f"\nClases detectadas: {class_names}")
    
    # Calcular pesos de clases para balancear
    class_weights = {}
    total_samples = train_generator.samples
    
    # Calcular pesos de clases para entrenamiento balanceado
    for class_name, class_idx in train_generator.class_indices.items():
        # Contar ejemplos de esta clase
        class_dir = os.path.join(train_dir, class_name)
        n_samples = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        
        # Calcular peso
        weight = total_samples / (len(class_names) * n_samples)
        class_weights[class_idx] = weight
        
        print(f"Clase '{class_name}': {n_samples} ejemplos, peso: {weight:.2f}")
    
    print("\n[2/5] Creando modelo...")
    
    # Crear modelo base (MobileNetV2)
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar el modelo base
    base_model.trainable = False
    
    # Crear modelo completo
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Resumen del modelo
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODELS_DIR, 'simple_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    print("\n[3/5] Entrenando modelo...")
    
    # Entrenar modelo
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Guardar modelo final
    model_path = os.path.join(config.MODELS_DIR, 'simple_model_final.h5')
    model.save(model_path)
    print(f"\nModelo guardado en: {model_path}")
    
    # Actualizar config.py con el nuevo modelo
    with open("config.py", "r") as f:
        content = f.read()
    
    content = content.replace(
        f'MEJOR_MODELO = "{config.MEJOR_MODELO}"',
        'MEJOR_MODELO = "simple_model_best.h5"'
    )
    
    with open("config.py", "w") as f:
        f.write(content)
    
    print("\n[4/5] Visualizando resultados de entrenamiento...")
    
    # Graficar historia de entrenamiento
    plt.figure(figsize=(12, 4))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'entrenamiento_simple.png'))
    
    print("\n[5/5] Evaluando modelo en datos de prueba...")
    
    # Evaluar modelo
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Precisión en conjunto de prueba: {test_acc:.4f}')
    print(f'Pérdida en conjunto de prueba: {test_loss:.4f}')
    
    # Matriz de confusión
    print("\nGenerando matriz de confusión...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.title('Matriz de Confusión')
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'matriz_confusion_simple.png'))
    
    # Calcular métricas
    from sklearn.metrics import classification_report
    print("\nInforme de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n" + "=" * 50)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 50)
    
    return model, history, train_generator, validation_generator, test_generator

if __name__ == "__main__":
    train_simple_model()