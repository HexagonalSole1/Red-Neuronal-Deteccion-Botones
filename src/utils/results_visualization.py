"""
Módulo para visualización avanzada de resultados del modelo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.preprocessing import image
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
from tensorflow.keras.models import Model
import matplotlib.cm as cm

# Importar configuración
import config

def plot_kfold_results(kfold_metrics, output_dir):
    """
    Visualiza los resultados de validación cruzada K-fold
    
    Args:
        kfold_metrics: Diccionario con métricas de cada fold
        output_dir: Directorio donde guardar las visualizaciones
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraer métricas
    folds = len(kfold_metrics['accuracy'])
    fold_nums = np.arange(1, folds + 1)
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Subplot para accuracy
    plt.subplot(2, 1, 1)
    plt.plot(fold_nums, kfold_metrics['accuracy'], 'o-', label='Accuracy')
    plt.axhline(y=np.mean(kfold_metrics['accuracy']), color='r', linestyle='--', 
                label=f'Promedio: {np.mean(kfold_metrics["accuracy"]):.4f}')
    
    # Añadir desviación estándar
    plt.fill_between(fold_nums, 
                     np.mean(kfold_metrics['accuracy']) - np.std(kfold_metrics['accuracy']), 
                     np.mean(kfold_metrics['accuracy']) + np.std(kfold_metrics['accuracy']), 
                     color='r', alpha=0.1)
    
    plt.title('Precisión por Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(fold_nums)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Subplot para loss
    plt.subplot(2, 1, 2)
    plt.plot(fold_nums, kfold_metrics['loss'], 'o-', label='Loss')
    plt.axhline(y=np.mean(kfold_metrics['loss']), color='r', linestyle='--', 
                label=f'Promedio: {np.mean(kfold_metrics["loss"]):.4f}')
    
    # Añadir desviación estándar
    plt.fill_between(fold_nums, 
                     np.mean(kfold_metrics['loss']) - np.std(kfold_metrics['loss']), 
                     np.mean(kfold_metrics['loss']) + np.std(kfold_metrics['loss']), 
                     color='r', alpha=0.1)
    
    plt.title('Pérdida por Fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.xticks(fold_nums)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kfold_results.png'))
    plt.close()
    
    # Crear tabla comparativa
    fig, ax = plt.figure(figsize=(10, 4)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    # Crear datos para la tabla
    data = []
    for i in range(folds):
        data.append([f'Fold {i+1}', 
                     f'{kfold_metrics["accuracy"][i]:.4f}', 
                     f'{kfold_metrics["loss"][i]:.4f}'])
    
    data.append(['Promedio', 
                 f'{np.mean(kfold_metrics["accuracy"]):.4f}', 
                 f'{np.mean(kfold_metrics["loss"]):.4f}'])
    
    data.append(['Desv. Estándar', 
                 f'{np.std(kfold_metrics["accuracy"]):.4f}', 
                 f'{np.std(kfold_metrics["loss"]):.4f}'])
    
    ax.table(cellText=data, 
             colLabels=['Fold', 'Accuracy', 'Loss'],
             loc='center', cellLoc='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kfold_table.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, title='Matriz de Confusión'):
    """
    Genera y guarda una matriz de confusión mejorada
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        output_dir: Directorio donde guardar la visualización
        title: Título para la matriz
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar matriz para visualización de porcentajes
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Configurar figura
    plt.figure(figsize=(10, 8))
    
    # Crear colormap personalizado
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Crear heatmap con valores absolutos y porcentajes
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                square=True, linewidths=0.5, cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    
    # Añadir porcentajes como texto adicional
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'{cm_norm[i, j]:.1%}', 
                     ha='center', va='center', color='black', fontsize=9)
    
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.title(title)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_detailed.png'), dpi=200)
    plt.close()
    
    # Imprimir informe de clasificación para análisis adicional
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Guardar informe
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Transformar el informe en una tabla visual
    # Parsear el informe para extraer los valores
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:  # Excluir encabezados y líneas finales
        if line.strip():
            row_data = line.strip().split()
            if len(row_data) == 5:  # Asumiendo formato clásico: class precision recall f1 support
                class_name = row_data[0]
                precision = float(row_data[1])
                recall = float(row_data[2])
                f1 = float(row_data[3])
                support = int(row_data[4])
                report_data.append([class_name, precision, recall, f1, support])
    
    # Convertir a DataFrame para facilitar la visualización
    df_report = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # Visualizar como tabla
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_report.values, 
                     colLabels=df_report.columns,
                     loc='center', 
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title('Informe de Clasificación')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_table.png'), dpi=200)
    plt.close()
    
    return cm, report

def visualize_predictions(model, test_generator, num_samples=8, output_dir=None):
    """
    Visualiza predicciones del modelo en muestras aleatorias
    
    Args:
        model: Modelo entrenado
        test_generator: Generador de datos de prueba
        num_samples: Número de muestras a visualizar
        output_dir: Directorio donde guardar las visualizaciones
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Obtener una muestra aleatoria de imágenes
    test_generator.reset()
    x_batch, y_batch = next(test_generator)
    
    # Si hay menos imágenes que num_samples, ajustar
    num_samples = min(num_samples, len(x_batch))
    
    # Hacer predicciones
    predictions = model.predict(x_batch[:num_samples])
    
    # Configurar cuadrícula
    rows = int(np.ceil(num_samples / 4))
    cols = min(4, num_samples)
    
    fig = plt.figure(figsize=(16, 4 * rows))
    
    # Obtener nombres de clases
    class_names = list(test_generator.class_indices.keys())
    
    # Visualizar cada muestra
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Mostrar imagen
        ax.imshow(x_batch[i])
        
        # Etiqueta real
        true_label = np.argmax(y_batch[i])
        true_label_name = class_names[true_label]
        
        # Etiqueta predicha
        pred_label = np.argmax(predictions[i])
        pred_label_name = class_names[pred_label]
        
        # Decidir color según si la predicción es correcta
        color = 'green' if true_label == pred_label else 'red'
        
        # Añadir título con información de la predicción
        ax.set_title(f'Real: {true_label_name}\nPred: {pred_label_name}', color=color)
        
        # Añadir barras de confianza
        for j, score in enumerate(predictions[i]):
            plt.text(x=10, y=150 + j*20, s=f'{class_names[j]}: {score:.2f}', 
                     transform=ax.transData, fontsize=9, 
                     color='black' if j == pred_label else 'gray')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=200)
        plt.close()
    else:
        plt.show()

def generate_gradcam_heatmap(model, img_array, layer_name='Conv_1', pred_index=None):
    """
    Genera un mapa de calor Grad-CAM para visualizar qué partes de la imagen
    son más importantes para la predicción
    
    Args:
        model: Modelo entrenado
        img_array: Array de la imagen (batch de 1)
        layer_name: Nombre de la capa para generar el Grad-CAM
        pred_index: Índice de la clase predicha (si None, se usa la de mayor probabilidad)
        
    Returns:
        Mapa de calor normalizado
    """
    # Crear modelo para Grad-CAM
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Computar gradiente del score de la clase objetivo con respecto a la capa especificada
    with tf.GradientTape() as tape:
        # Ejecución forward
        conv_outputs, predictions = grad_model(img_array)
        
        # Usar la predicción con mayor probabilidad si no se especifica
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Obtener el score de la clase objetivo
        class_channel = predictions[:, pred_index]
    
    # Gradiente del score de clase con respecto a la salida de la capa convolucional
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector de importancia: pesos globales de los mapas de características
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Ponderar mapas de activación con los gradientes
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalizar mapa de calor para visualización
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def apply_gradcam(model, img_path, layer_name='Conv_1', output_dir=None):
    """
    Aplica Grad-CAM a una imagen y guarda o muestra la visualización
    
    Args:
        model: Modelo entrenado
        img_path: Ruta a la imagen
        layer_name: Nombre de la capa para generar el Grad-CAM
        output_dir: Directorio donde guardar la visualización
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Cargar y preprocesar imagen
    img = image.load_img(img_path, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predecir clase
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    
    # Nombres de clases
    class_names = config.CLASES
    
    # Generar Grad-CAM
    try:
        heatmap = generate_gradcam_heatmap(model, img_array, layer_name, pred_class)
    except ValueError:
        # Si no se encuentra la capa, intentar con la última capa convolucional
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        
        if layer_name:
            print(f"Usando capa alternativa: {layer_name}")
            heatmap = generate_gradcam_heatmap(model, img_array, layer_name, pred_class)
        else:
            print("No se pudo encontrar una capa convolucional para Grad-CAM")
            return
    
    # Convertir imagen a RGB
    img = cv2.imread(img_path)
    img = cv2.resize(img, (config.ANCHO_IMAGEN, config.ALTURA_IMAGEN))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar heatmap al tamaño de la imagen
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convertir heatmap a RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer heatmap en la imagen original con un factor de superposición
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Visualizar
    plt.figure(figsize=(12, 4))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')
    
    # Superposición
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f'Predicción: {class_names[pred_class]}\n(Confianza: {preds[0][pred_class]:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        img_name = os.path.basename(img_path).split('.')[0]
        plt.savefig(os.path.join(output_dir, f'gradcam_{img_name}.png'), dpi=200)
        plt.close()
    else:
        plt.show()

def create_results_collage(model, image_paths, output_path, include_gradcam=True):
    """
    Crea un collage con predicciones y visualizaciones para múltiples imágenes
    
    Args:
        model: Modelo entrenado
        image_paths: Lista de rutas a imágenes
        output_path: Ruta para guardar el collage
        include_gradcam: Si se debe incluir la visualización Grad-CAM
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determinar la última capa convolucional para Grad-CAM
    gradcam_layer = None
    if include_gradcam:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                gradcam_layer = layer.name
                break
    
    # Configurar cuadrícula
    n_images = len(image_paths)
    cols = min(4, n_images)
    rows = int(np.ceil(n_images / cols))
    
    fig_width = 5 * cols
    fig_height = 6 * rows
    
    # Crear figura
    plt.figure(figsize=(fig_width, fig_height))
    
    for i, img_path in enumerate(image_paths):
        # Cargar imagen
        try:
            img = image.load_img(img_path, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0) / 255.0
            
            # Hacer predicción
            preds = model.predict(img_batch)
            pred_class = np.argmax(preds[0])
            confidence = preds[0][pred_class]
            
            # Nombre de la clase predicha
            class_name = config.CLASES[pred_class]
            
            # Crear subplot
            plt.subplot(rows, cols, i + 1)
            
            if include_gradcam and gradcam_layer:
                # Generar Grad-CAM
                try:
                    heatmap = generate_gradcam_heatmap(model, img_batch, gradcam_layer, pred_class)
                    
                    # Redimensionar heatmap al tamaño de la imagen
                    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
                    
                    # Convertir heatmap a RGB
                    heatmap_rgb = np.uint8(255 * heatmap)
                    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
                    
                    # Superponer heatmap en la imagen original
                    alpha = 0.35
                    superimposed = heatmap_rgb * alpha + img_array * (1 - alpha)
                    superimposed = np.clip(superimposed, 0, 255).astype('uint8')
                    
                    # Mostrar imagen superpuesta
                    plt.imshow(superimposed)
                except Exception as e:
                    print(f"Error generando Grad-CAM para {img_path}: {e}")
                    plt.imshow(img)
            else:
                # Mostrar imagen original
                plt.imshow(img)
            
            # Añadir título con predicción y confianza
            plt.title(f"Pred: {class_name}\nConf: {confidence:.2f}")
            plt.axis('off')
            
            # Añadir barras de confianza para todas las clases
            y_offset = img_array.shape[0] + 5
            max_bar_width = img_array.shape[1] - 20
            
            for j, score in enumerate(preds[0]):
                # Calcular largo de barra
                bar_width = max_bar_width * score
                
                # Color de la barra
                bar_color = 'green' if j == pred_class else 'gray'
                
                # Dibujar barra
                plt.text(10, y_offset + j*15, config.CLASES[j], fontsize=8, ha='left')
                plt.hlines(y=y_offset + j*15 + 5, xmin=10, xmax=10 + bar_width, 
                           colors=bar_color, linewidth=6, alpha=0.7)
                plt.text(10 + bar_width + 5, y_offset + j*15 + 5, f'{score:.2f}', 
                         fontsize=7, va='center')
            
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Collage guardado en: {output_path}")

def main():
    """Función principal de demostración"""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Cargar modelo
    model_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")
    else:
        print(f"No se encontró el modelo en: {model_path}")
        return
    
    # Directorio de salida
    output_dir = os.path.join(config.OUTPUT_DIR, 'visualizaciones')
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos de prueba
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_dir = os.path.join(config.DATA_DIR, 'prueba')
    if os.path.exists(test_dir):
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN),
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Visualizar algunas predicciones
        visualize_predictions(model, test_generator, num_samples=8, 
                             output_dir=output_dir)
        
        # Evaluar y generar matriz de confusión
        test_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
        predictions = model.predict(test_generator, steps=test_steps)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes[:len(y_pred)]
        
        plot_confusion_matrix(y_true, y_pred, config.CLASES, output_dir)
        
        # Buscar algunas imágenes para Grad-CAM
        for class_name in config.CLASES:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f)) and 
                              f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    # Tomar la primera imagen encontrada
                    img_path = image_files[0]
                    apply_gradcam(model, img_path, output_dir=output_dir)
        
        # Crear collage de resultados
        # Seleccionar algunas imágenes aleatorias para el collage
        all_images = []
        for class_name in config.CLASES:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f)) and 
                              f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Tomar hasta 2 imágenes de cada clase
                if image_files:
                    selected = np.random.choice(image_files, size=min(2, len(image_files)), replace=False)
                    all_images.extend(selected)
        
        if all_images:
            create_results_collage(model, all_images, 
                                 os.path.join(output_dir, 'results_collage.png'))
    else:
        print(f"No se encontró el directorio de prueba: {test_dir}")

if __name__ == "__main__":
    main()