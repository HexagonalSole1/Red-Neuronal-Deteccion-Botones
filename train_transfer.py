from train.train_transfer_model import train_transfer_model
import argparse

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Entrenar clasificador de botones con transfer learning')
parser.add_argument('--fine-tuning', action='store_true', help='Activar fine-tuning en el modelo base')
parser.add_argument('--no-pesos', action='store_true', help='Desactivar el uso de pesos de clase')
parser.add_argument('--modelo', type=str, choices=['mobilenet', 'resnet', 'efficientnet'], default='mobilenet',
                    help='Backbone a utilizar: mobilenet (default), resnet o efficientnet')

# Parsear argumentos
args = parser.parse_args()

# Entrenar el modelo con los parámetros indicados
train_transfer_model(
    fine_tuning=args.fine_tuning,
    usar_pesos_clase=not args.no_pesos,
    backbone=args.modelo
)
