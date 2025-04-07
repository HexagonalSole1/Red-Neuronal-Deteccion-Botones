
"""
Script para implementar validación cruzada K-fold en el clasificador de botones
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
from datetime import datetime
import argparse
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Importar módulos propios
import config
from models.model_transfer import crear_modelo_transfer, crear_modelo_transfer_fine_tuning
from models.modelo_cnn import crear_modelo_cnn, crear_modelo_cnn_basico
from src.utils.visualization import plot_training_history
from src.utils.heic_converter import prepare_image_directories

# Configuración para reproducibilidad
def set_seed(seed=42):
    """Establece semillas para reproducibilidad"""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def load_dataset_filenames():
    """
    Carga los nombres de archivo del dataset para validación cruzada
    
    Returns:
        Tupla de (filenames, labels) para cada imagen del dataset
    """
    train_dir = os.path.join(config.DATA_DIR, 'entrenamiento')
    
    # Verificar que el directorio existe
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"El directorio de entrenamiento no existe: {train_dir}")
    
    filenames = []
    labels = []
    label_to_index = {}
    
    # Crear índices para las clases
    for i, class_name in enumerate(config.CLASES):
        label_to_index[class_name] = i
    
    # Recorrer directorios de clases
    for class_name in config.CLASES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            # Obtener archivos de imágenes
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in valid_extensions:
                image_files = glob.glob(os.path.join(class_dir, f'*{ext}'))
                for img_path in image_files:
                    filenames.append(img_path)
                    labels.append(label_to_index[class_name])
    
    return np.array(filenames), np.array(labels), label_to_index

def create_image_generator(preprocessing_function=None):
    """
    Crea un generador de imágenes con data augmentation
    
    Args:
        preprocessing_function: Función de preprocesamiento para el generador
        
    Returns:
        Generador de imágenes configurado
    """
    # Configurar generador para entrenamiento con augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=config.DATA_AUGMENTATION['rotation_range'],
        width_shift_range=config.DATA_AUGMENTATION['width_shift_range'],
        height_shift_range=config.DATA_AUGMENTATION['height_shift_range'],
        shear_range=config.DATA_AUGMENTATION['shear_range'],
        zoom_range=config.DATA_AUGMENTATION['zoom_range'],
        horizontal_flip=config.DATA_AUGMENTATION['horizontal_flip'],
        vertical_flip=config.DATA_AUGMENTATION['vertical_flip'],
        brightness_range=config.DATA_AUGMENTATION['brightness_range'],
        fill_mode=config.DATA_AUGMENTATION['fill_mode']
    )
    
    # Generador para validación sin augmentation
    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    
    return train_datagen, valid_datagen

def create_model(model_type='transfer', fine_tuning=False):
    """
    Crea el modelo según el tipo especificado
    
    Args:
        model_type: Tipo de modelo ('basic', 'cnn', o 'transfer')
        fine_tuning: Si se debe aplicar fine-tuning al modelo de transfer learning
        
    Returns:
        Modelo construido
    """
    if model_type == 'basic':
        return crear_modelo_cnn_basico(config)
    elif model_type == 'cnn':
        return crear_modelo_cnn(config)
    elif model_type == 'transfer':
        if fine_tuning:
            return crear_modelo_transfer_fine_tuning(config)
        else:
            return crear_modelo_transfer(config)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

def train_with_kfold_validation(k_folds=5, model_type='transfer', fine_tuning=False, 
                               batch_size=None, epochs=None, use_class_weights=True):
    """
    Entrena el modelo usando validación cruzada K-fold
    
    Args:
        k_folds: Número de particiones para validación cruzada
        model_type: Tipo de modelo a entrenar
        fine_tuning: Si se debe aplicar fine-tuning al modelo de transfer learning
        batch_size: Tamaño de batch para entrenamiento
        epochs: Número de épocas para entrenamiento
        use_class_weights: Si se deben aplicar pesos de clase
        
    Returns:
        Diccionario con resultados de validación cruzada
    """
    print("=" * 50)
    print(f"ENTRENAMIENTO CON VALIDACIÓN CRUZADA {k_folds}-FOLD")
    print("=" * 50)
    
    # Establecer valores por defecto
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    if epochs is None:
        epochs = config.EPOCAS
    
    # Semilla para reproducibilidad
    set_seed()
    
    # Crear directorios de salida
    kfold_dir = os.path.join(config.OUTPUT_DIR, f'kfold_{model_type}')
    os.makedirs(kfold_dir, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Preparar imágenes
    print("\n[1/4] Preparando imágenes...")
    prepare_image_directories(config)
    
    # Cargar nombres de archivos y etiquetas
    print("\n[2/4] Cargando dataset...")
    filenames, labels, label_to_index = load_dataset_filenames()
    
    # Crear generadores de imágenes
    print("\n[3/4] Configurando generadores de datos...")
    preprocessing_function = None
    if model_type == 'transfer':
        # Usar preprocesamiento específico para MobileNetV2
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
    
    train_datagen, valid_datagen = create_image_generator(preprocessing_function)
    
    # Calcular pesos de clase si es necesario
    class_weights = None
    if use_class_weights:
        # Calcular pesos inversamente proporcionales a la frecuencia de clase
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_arr)}
        
        print("\nPesos de clase calculados:")
        for class_idx, weight in class_weights.items():
            for class_name, idx in label_to_index.items():
                if idx == class_idx:
                    print(f"  - Clase '{class_name}' (índice {class_idx}): peso {weight:.4f}")
    
    # Configurar K-fold cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Métricas para cada fold
    fold_metrics = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Para guardar los modelos de cada fold
    fold_models = []
    
    # Para guardar matriz de confusión combinada
    y_true_all = []
    y_pred_all = []
    
    # Ejecutar entrenamiento con K-fold
    print(f"\n[4/4] Iniciando entrenamiento con {k_folds}-fold cross validation...")
    
    fold_no = 1
    for train_idx, val_idx in kfold.split(filenames):
        print(f"\n{'='*20} Fold {fold_no}/{k_folds} {'='*20}")
        
        # Obtener nombres de archivo para este fold
        train_filenames, val_filenames = filenames[train_idx], filenames[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Crear generadores para este fold
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(config.DATA_DIR, 'entrenamiento'),
            target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN),
            batch_size=batch_size,
            class_mode='categorical',
            classes=config.CLASES,
            shuffle=True,
            subset=None
        )
        
        val_generator = valid_datagen.flow_from_directory(
            directory=os.path.join(config.DATA_DIR, 'entrenamiento'),
            target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN),
            batch_size=batch_size,
            class_mode='categorical',
            classes=config.CLASES,
            shuffle=False,
            subset=None
        )
        
        # Crear modelo para este fold
        model = create_model(model_type, fine_tuning)
        
        # Callbacks para este fold
        fold_output_dir = os.path.join(kfold_dir, f'fold_{fold_no}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            os.path.join(fold_output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        
        # Entrenamiento para este fold
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_idx) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_idx) // batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weights
        )
        
        # Evaluar el modelo en el conjunto de validación
        val_generator.reset()
        results = model.evaluate(val_generator)
        
        print(f"\nResultados para Fold {fold_no}:")
        print(f"  - Loss: {results[0]:.4f}")
        print(f"  - Accuracy: {results[1]:.4f}")
        
        # Guardar métricas
        fold_metrics['loss'].append(results[0])
        fold_metrics['accuracy'].append(results[1])
        fold_metrics['val_loss'].append(min(history.history['val_loss']))
        fold_metrics['val_accuracy'].append(max(history.history['val_accuracy']))
        
        # Guardar historia de entrenamiento
        plt.figure(figsize=(12, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Precisión del modelo - Fold {fold_no}')
        plt.ylabel('Precisión')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Pérdida del modelo - Fold {fold_no}')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_output_dir, 'training_history.png'))
        plt.close()
        
        # Obtener predicciones para matriz de confusión
        val_generator.reset()
        y_pred = model.predict(val_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Solo considerar hasta el número de muestras reales
        y_true = val_generator.classes[:len(y_pred_classes)]
        y_pred_classes = y_pred_classes[:len(y_true)]
        
        # Guardar para matriz de confusión combinada
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred_classes)
        
        # Crear y guardar matriz de confusión para este fold
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.CLASES, yticklabels=config.CLASES)
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.title(f'Matriz de Confusión - Fold {fold_no}')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Imprimir reporte de clasificación
        class_report = classification_report(y_true, y_pred_classes, target_names=config.CLASES)
        print("\nReporte de Clasificación:")
        print(class_report)
        
        # Guardar reporte
        with open(os.path.join(fold_output_dir, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
        
        # Guardar modelo para este fold
        model.save(os.path.join(fold_output_dir, 'final_model.h5'))
        fold_models.append(model)
        
        fold_no += 1
    
    # Calcular promedios
    avg_loss = np.mean(fold_metrics['loss'])
    avg_acc = np.mean(fold_metrics['accuracy'])
    std_loss = np.std(fold_metrics['loss'])
    std_acc = np.std(fold_metrics['accuracy'])
    
    print("\n" + "=" * 50)
    print("RESULTADOS DE VALIDACIÓN CRUZADA")
    print("=" * 50)
    print(f"Precisión promedio: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Pérdida promedio: {avg_loss:.4f} ± {std_loss:.4f}")
    
    # Crear matriz de confusión combinada
    from sklearn.metrics import confusion_matrix, classification_report
    cm_all = confusion_matrix(y_true_all, y_pred_all)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues',
               xticklabels=config.CLASES, yticklabels=config.CLASES)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.title('Matriz de Confusión Combinada')
    plt.tight_layout()
    plt.savefig(os.path.join(kfold_dir, 'combined_confusion_matrix.png'))
    
    # Generar gráfica de barras con resultados por fold
    plt.figure(figsize=(12, 6))
    x = np.arange(k_folds)
    width = 0.35
    
    plt.bar(x - width/2, fold_metrics['accuracy'], width, label='Accuracy')
    plt.bar(x + width/2, fold_metrics['loss'], width, label='Loss')
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Resultados por Fold')
    plt.xticks(x, [f'Fold {i+1}' for i in range(k_folds)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(kfold_dir, 'fold_results.png'))
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(k_folds)],
        'Accuracy': fold_metrics['accuracy'],
        'Loss': fold_metrics['loss'],
        'Val_Accuracy': fold_metrics['val_accuracy'],
        'Val_Loss': fold_metrics['val_loss']
    })
    
    results_df.loc[len(results_df)] = ['Promedio', avg_acc, avg_loss, 
                                      np.mean(fold_metrics['val_accuracy']), 
                                      np.mean(fold_metrics['val_loss'])]
    
    results_df.loc[len(results_df)] = ['Desv. Estándar', std_acc, std_loss,
                                      np.std(fold_metrics['val_accuracy']),
                                      np.std(fold_metrics['val_loss'])]
    
    results_df.to_csv(os.path.join(kfold_dir, 'kfold_results.csv'), index=False)
    
    # Guardar el mejor modelo como el modelo final
    best_fold_idx = np.argmax(fold_metrics['accuracy'])
    best_model = fold_models[best_fold_idx]
    
    # Nombre del modelo según tipo
    model_name_parts = []
    model_name_parts.append(model_type)
    if model_type == 'transfer' and fine_tuning:
        model_name_parts.append('fine_tuned')
    model_name_parts.append('kfold')
    model_name = '_'.join(model_name_parts)
    
    # Guardar mejor modelo
    best_model_path = os.path.join(config.MODELS_DIR, f'mejor_modelo_{model_name}.h5')
    best_model.save(best_model_path)
    
    # Actualizar el archivo config.py
    with open('config.py', 'r') as file:
        config_content = file.read()
    
    # Reemplazar la línea MEJOR_MODELO
    config_content = config_content.replace(
        f'MEJOR_MODELO = "{config.MEJOR_MODELO}"', 
        f'MEJOR_MODELO = "mejor_modelo_{model_name}.h5"'
    )
    
    # Guardar el archivo config actualizado
    with open('config.py', 'w') as file:
        file.write(config_content)
    
    print("\nValidación cruzada completada:")
    print(f"- Resultados guardados en: {kfold_dir}")
    print(f"- Mejor modelo guardado como: {best_model_path}")
    print(f"- Archivo config.py actualizado para usar el nuevo modelo por defecto")
    
    # Devolver resultados
    return {
        'avg_accuracy': avg_acc,
        'std_accuracy': std_acc,
        'avg_loss': avg_loss,
        'std_loss': std_loss,
        'fold_metrics': fold_metrics,
        'best_model': best_model,
        'best_model_path': best_model_path,
        'output_dir': kfold_dir
    }

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento con validación cruzada K-fold')
    parser.add_argument('--folds', type=int, default=5, 
                        help='Número de folds para validación cruzada (default: 5)')
    parser.add_argument('--model', type=str, default='transfer', 
                        choices=['basic', 'cnn', 'transfer'],
                        help='Tipo de modelo a entrenar (default: transfer)')
    parser.add_argument('--fine-tuning', action='store_true', 
                        help='Usar fine-tuning para modelo de transfer learning')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas para entrenamiento')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Tamaño de batch para entrenamiento')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='No usar pesos de clase para entrenamiento')
    
    args = parser.parcse_args()
    
    # Ejecutar entrenamiento con validación cruzada
    train_with_kfold_validation(
        k_folds=args.folds,
        model_type=args.model,
        fine_tuning=args.fine_tuning,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_class_weights=not args.no_class_weights
    )

if __name__ == "__main__":
    main()