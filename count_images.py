import os
from collections import defaultdict

EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')  # Puedes modificar según lo que uses

def contar_imagenes_por_clase(data_dir):
    """
    Recorre las carpetas del dataset y cuenta cuántas imágenes válidas hay por clase.

    Args:
        data_dir (str): Ruta a la carpeta principal que contiene subcarpetas por clase.

    Returns:
        dict: Diccionario con el conteo por clase.
    """
    conteo = defaultdict(int)
    
    for subset in ['entrenamiento', 'validacion', 'prueba']:
        subset_path = os.path.join(data_dir, subset)
        print(f"\n📁 Subconjunto: {subset}")
        
        if not os.path.exists(subset_path):
            print(f"  ⚠️ No existe el directorio: {subset_path}")
            continue

        for clase in os.listdir(subset_path):
            clase_path = os.path.join(subset_path, clase)
            if os.path.isdir(clase_path):
                n_imgs = len([
                    f for f in os.listdir(clase_path)
                    if os.path.isfile(os.path.join(clase_path, f)) and f.lower().endswith(EXTENSIONES_VALIDAS)
                ])
                conteo[(subset, clase)] = n_imgs
                print(f"  - Clase '{clase}': {n_imgs} imágenes válidas")
    
    return conteo

# Ejecución directa para revisión rápida
if __name__ == "__main__":
    import config
    conteo = contar_imagenes_por_clase(config.DATA_DIR)
    print("\n✅ Conteo final de imágenes válidas por clase:")
    for (subset, clase), cantidad in conteo.items():
        print(f"[{subset}] {clase}: {cantidad}")
