"""
Funciones para visualización de resultados y matriz de confusión
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def plot_training_history(history, output_dir):
    """
    Visualiza el historial de entrenamiento
    
    Args:
        history: Objeto history devuelto por model.fit
        output_dir: Directorio donde guardar la visualización
    """
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
    
    # Guardar figura
    plt.savefig(os.path.join(output_dir, 'historia_entrenamiento.png'))
    plt.close()

def generar_matriz_confusion(model, test_generator, config, output_dir):
    """
    Genera y visualiza la matriz de confusión
    
    Args:
        model: Modelo entrenado
        test_generator: Generador de datos de prueba
        config: Módulo de configuración
        output_dir: Directorio donde guardar la visualización
        
    Returns:
        cm: Matriz de confusión
    """
    # Obtener predicciones
    test_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    predictions = model.predict(test_generator, steps=test_steps)
    y_pred = np.argmax(predictions, axis=1)
    
    # Obtener etiquetas reales (limitado al número de predicciones)
    y_true = test_generator.classes[:len(y_pred)]
    
    # Obtener nombres de clases
    class_names = list(test_generator.class_indices.keys())
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar matriz para mejor visualización
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.title('Matriz de Confusión')
    
    # Guardar figura
    plt.savefig(os.path.join(output_dir, 'matriz_confusion.png'))
    plt.close()
    
    # Imprimir informe de clasificación
    print("\nInforme de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm