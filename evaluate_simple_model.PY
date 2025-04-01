"""
Script para evaluar el modelo simple entrenado
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importar configuración
import config

def evaluar_modelo_simple(modelo_path=None):
    """
    Evalúa el modelo simple
    
    Args:
        modelo_path: Ruta al modelo. Si es None, se usa el mejor modelo guardado.
    """
    print("=" * 50)
    print("EVALUACIÓN DEL MODELO SIMPLE")
    print("=" * 50)
    
    # Determinar qué modelo usar
    if modelo_path is None:
        modelo_path = os.path.join(config.MODELS_DIR, 'simple_model_best.h5')
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"El modelo no existe en la ruta: {modelo_path}")
    
    # Directorio de prueba
    test_dir = os.path.join(config.DATA_DIR, 'prueba')
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"El directorio de prueba no existe: {test_dir}")
    
    # Preprocesamiento para MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Generador para prueba
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Crear generador de prueba
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Importante para mantener orden para matriz de confusión
    )
    
    # Obtener nombres de clases
    class_names = list(test_generator.class_indices.keys())
    
    print(f"\n[1/3] Cargando modelo desde: {modelo_path}")
    model = load_model(modelo_path)
    
    print("\n[2/3] Evaluando modelo...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Precisión en conjunto de prueba: {test_acc:.4f}')
    print(f'Pérdida en conjunto de prueba: {test_loss:.4f}')
    
    print("\n[3/3] Generando matriz de confusión...")
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
    
    # Guardar figura
    output_path = os.path.join(config.OUTPUT_DIR, 'matriz_confusion_eval.png')
    plt.savefig(output_path)
    plt.close()
    
    # Imprimir informe de clasificación
    print("\nInforme de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print("\n" + "=" * 50)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 50)
    print(f"Matriz de confusión guardada en: {output_path}")
    
    return test_acc, test_loss, cm

if __name__ == "__main__":
    evaluar_modelo_simple()