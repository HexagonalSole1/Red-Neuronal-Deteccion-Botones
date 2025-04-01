"""
Script para evaluar el modelo entrenado y generar la matriz de confusión
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Importar módulos propios
import config
from data.preprocesar_datos import crear_generadores_datos
from src.utils.visualization import generar_matriz_confusion

def evaluar_modelo(modelo_path=None):
    """
    Evalúa el modelo y genera la matriz de confusión
    
    Args:
        modelo_path: Ruta al modelo. Si es None, se usa el mejor modelo guardado.
    """
    print("=" * 50)
    print("EVALUACIÓN DEL CLASIFICADOR DE BOTONES")
    print("=" * 50)
    
    # Determinar qué modelo usar
    if modelo_path is None:
        modelo_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
    
    # Crear directorios si no existen
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Cargar los datos (solo necesitamos test_generator)
    print("\n[1/3] Cargando datos de prueba...")
    _, _, test_generator = crear_generadores_datos(config)
    
    # Cargar el modelo
    print(f"\n[2/3] Cargando modelo desde: {modelo_path}")
    model = load_model(modelo_path)
    
    # Evaluar el modelo
    print("\n[2/3] Evaluando modelo...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Precisión en conjunto de prueba: {test_acc:.4f}')
    print(f'Pérdida en conjunto de prueba: {test_loss:.4f}')
    
    # Generar matriz de confusión
    print("\n[3/3] Generando matriz de confusión...")
    cm = generar_matriz_confusion(model, test_generator, config, config.OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 50)
    print(f"Matriz de confusión guardada en: {os.path.join(config.OUTPUT_DIR, 'matriz_confusion.png')}")
    
    return test_acc, test_loss, cm

if __name__ == "__main__":
    evaluar_modelo()