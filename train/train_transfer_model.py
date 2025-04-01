import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos propios
import config
from data.preprocesar_datos import crear_generadores_datos
from models.model_transfer import crear_modelo_transfer
from src.utils.visualization import plot_training_history
from src.utils.heic_converter import prepare_image_directories


def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def calcular_pesos_clase(train_generator):
    class_indices = train_generator.class_indices
    class_counts = {}

    for clase in class_indices.keys():
        ruta_clase = os.path.join(config.DATA_DIR, "entrenamiento", clase)
        class_counts[clase] = len(os.listdir(ruta_clase))

    total_samples = sum(class_counts.values())
    class_weights = {}

    print("\nDistribución de clases:")
    for i, (clase, count) in enumerate(class_counts.items()):
        weight = total_samples / (len(class_indices) * count)
        class_weights[i] = weight
        print(f"  - Clase '{clase}': {count} ejemplos ({count/total_samples*100:.2f}%), peso {weight:.2f}")

    return class_weights


def preparar_datos_transfer_learning():
    return {
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': False,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    }


def crear_callbacks(config, modelo_nombre):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODELS_DIR, f"mejor_modelo_{modelo_nombre}.h5"),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.OUTPUT_DIR, 'logs'), histogram_freq=1,
            write_graph=True, write_images=True
        )
    ]


def evaluar_modelo(model, test_generator, output_dir):
    print("\n[6/6] Evaluando modelo...")
    loss, acc = model.evaluate(test_generator)
    print(f"Precisión: {acc*100:.2f}% | Pérdida: {loss:.4f}")

    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matriz_confusion.png"))
    plt.close()


def guardar_configuracion(params, output_dir):
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(params, f, indent=4)


def get_preprocessing_function(backbone):
    if backbone == 'mobilenet':
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone == 'resnet':
        return tf.keras.applications.resnet.preprocess_input
    elif backbone == 'efficientnet':
        return tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError("Backbone no soportado. Usa 'mobilenet', 'resnet' o 'efficientnet'.")


def train_transfer_model(fine_tuning=False, usar_pesos_clase=True, backbone='mobilenet'):
    print("=" * 50)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE BOTONES CON TRANSFER LEARNING")
    print("=" * 50)

    set_seed()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/6] Preparando imágenes...")
    prepare_image_directories(config)

    data_augmentation = preparar_datos_transfer_learning()
    config.DATA_AUGMENTATION = data_augmentation

    print("\n[2/6] Cargando datos...")
    preprocess_fn = get_preprocessing_function(backbone)
    train_generator, val_generator, test_generator = crear_generadores_datos(
        config,
        preprocessing_function=preprocess_fn
    )

    print("\n[3/6] Creando modelo...")
    modelo_nombre = f"{backbone}_{'fine_tuned' if fine_tuning else 'frozen'}"
    model = crear_modelo_transfer(config, backbone_name=backbone, fine_tuning=fine_tuning)

    callbacks = crear_callbacks(config, modelo_nombre)

    class_weights = calcular_pesos_clase(train_generator) if usar_pesos_clase else None

    print("\n[4/6] Entrenando modelo...")
    start = datetime.now()
    epocas = config.EPOCAS * 2 if fine_tuning else config.EPOCAS

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // config.BATCH_SIZE,
        epochs=epocas,
        callbacks=callbacks,
        class_weight=class_weights
    )

    duration = datetime.now() - start
    model.save(os.path.join(config.MODELS_DIR, f"modelo_final_{modelo_nombre}.h5"))

    with open("config.py", "r") as f:
        content = f.read()
    content = content.replace(
        f'MEJOR_MODELO = "{config.MEJOR_MODELO}"',
        f'MEJOR_MODELO = "mejor_modelo_{modelo_nombre}.h5"'
    )
    with open("config.py", "w") as f:
        f.write(content)

    print("\n[5/6] Visualizando resultados...")
    plot_training_history(history, config.OUTPUT_DIR)

    evaluar_modelo(model, test_generator, config.OUTPUT_DIR)

    guardar_configuracion({
        "fine_tuning": fine_tuning,
        "epocas": epocas,
        "batch_size": config.BATCH_SIZE,
        "usar_pesos_clase": usar_pesos_clase,
        "modelo": modelo_nombre,
        "backbone": backbone,
        "datetime": str(datetime.now())
    }, config.OUTPUT_DIR)

    print("\n" + "=" * 50)
    print(f"ENTRENAMIENTO COMPLETADO EN: {duration}")
    print(f"Modelo guardado en: modelo_final_{modelo_nombre}.h5")
    print(f"Mejor modelo: mejor_modelo_{modelo_nombre}.h5")
    print("=" * 50)

    return model, history, train_generator, val_generator, test_generator
