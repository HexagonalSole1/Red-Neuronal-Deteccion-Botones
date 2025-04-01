# Clasificador de Botones mediante Imágenes

Este proyecto implementa un clasificador de botones mediante imágenes utilizando redes neuronales convolucionales (CNN) en TensorFlow.

## Requisitos

- Python 3.13.1 o superior
- TensorFlow 2.15.0 o superior
- Otras dependencias listadas en requirements.txt

## Estructura del Proyecto
clasificador-botones/
│
├── data/                      # Directorio para datos
│   ├── entrenamiento/         # Imágenes de entrenamiento
│   └── prueba/                # Imágenes de prueba
│
├── models/                    # Modelos guardados
├── output/                    # Visualizaciones y resultados
├── src/                       # Código fuente
│   ├── data/                  # Procesamiento de datos
│   ├── models/                # Definición de modelos
│   └── utils/                 # Utilidades y visualización
│
├── config.py                  # Configuración global
├── train.py                   # Script de entrenamiento
├── evaluate.py                # Script de evaluación
├── predict.py                 # Script para predicciones
└── README.md                  # Este archivo