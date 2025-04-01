"""
Script simplificado para clasificación de botones con interfaz gráfica Tkinter
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox

# Silenciar advertencias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Importar módulos propios
import config

class SimpleButtonClassifier:
    def __init__(self, root):
        # Configuración de la ventana principal
        self.root = root
        self.root.title("Clasificador de Botones")
        self.root.geometry("800x600")
        
        # Variables
        self.modelo = None
        self.ruta_imagen = None
        
        # Crear interfaz básica
        self.crear_interfaz()
        
        # Cargar el modelo
        self.modelo = self.cargar_modelo()
        
    def crear_interfaz(self):
        """Crear una interfaz minimalista"""
        # Frame principal
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        tk.Label(frame, text="CLASIFICADOR DE BOTONES", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Botones
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_select = tk.Button(btn_frame, text="Seleccionar Imagen", command=self.seleccionar_imagen)
        self.btn_select.pack(side=tk.LEFT, padx=10)
        
        self.btn_predict = tk.Button(btn_frame, text="Predecir", command=self.predecir, state=tk.DISABLED)
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        # Etiqueta para mostrar la ruta de la imagen
        self.lbl_path = tk.Label(frame, text="No se ha seleccionado ninguna imagen", wraplength=700)
        self.lbl_path.pack(fill=tk.X, pady=10)
        
        # Frame para resultados
        self.result_frame = tk.Frame(frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Etiqueta inicial
        self.lbl_info = tk.Label(self.result_frame, text="Selecciona una imagen y haz clic en Predecir")
        self.lbl_info.pack(pady=20)
        
        # Barra de estado
        self.lbl_status = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)
    
    def cargar_modelo(self):
        """Cargar el modelo de clasificación"""
        try:
            # Verificar si el modelo existe
            modelo_path = os.path.join(config.MODELS_DIR, config.MEJOR_MODELO)
            if not os.path.exists(modelo_path):
                self.lbl_status.config(text=f"Error: No se encontró el modelo en {modelo_path}")
                messagebox.showerror("Error", f"No se encontró el modelo en:\n{modelo_path}")
                return None
            
            # Cargar el modelo
            self.lbl_status.config(text=f"Cargando modelo desde {modelo_path}...")
            self.root.update()
            
            modelo = load_model(modelo_path)
            
            self.lbl_status.config(text="Modelo cargado correctamente")
            return modelo
            
        except Exception as e:
            self.lbl_status.config(text=f"Error al cargar el modelo: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar el modelo:\n{str(e)}")
            return None
    
    def seleccionar_imagen(self):
        """Seleccionar una imagen para clasificar"""
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*")
        ]
        
        ruta = filedialog.askopenfilename(title="Seleccionar una imagen", filetypes=filetypes)
        
        if ruta:
            self.ruta_imagen = ruta
            self.lbl_path.config(text=f"Imagen seleccionada: {ruta}")
            self.btn_predict.config(state=tk.NORMAL)
            self.lbl_status.config(text=f"Imagen seleccionada: {os.path.basename(ruta)}")
    
    def predecir(self):
        """Realizar predicción con la imagen seleccionada"""
        if not self.modelo:
            messagebox.showerror("Error", "El modelo no está cargado correctamente")
            return
        
        if not self.ruta_imagen or not os.path.exists(self.ruta_imagen):
            messagebox.showerror("Error", "Selecciona una imagen válida primero")
            return
        
        try:
            # Actualizar estado
            self.lbl_status.config(text="Procesando imagen...")
            self.root.update()
            
            # Cargar y preprocesar imagen
            img = image.load_img(self.ruta_imagen, target_size=(config.ALTURA_IMAGEN, config.ANCHO_IMAGEN))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Realizar predicción
            prediccion = self.modelo.predict(img_array, verbose=0)
            clase_id = np.argmax(prediccion, axis=1)[0]
            
            # Mostrar resultados
            self.mostrar_resultados(img, clase_id, prediccion[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la predicción:\n{str(e)}")
            self.lbl_status.config(text=f"Error: {str(e)}")
    
    def mostrar_resultados(self, img, clase_id, probabilidades):
        """Mostrar resultados de la predicción"""
        # Limpiar frame de resultados
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Crear figura de matplotlib
        fig = Figure(figsize=(8, 4), dpi=100)
        
        # Subplot para la imagen
        ax1 = fig.add_subplot(121)
        ax1.imshow(img)
        ax1.set_title(f"Clase: {config.CLASES[clase_id]}")
        ax1.axis('off')
        
        # Subplot para las probabilidades
        ax2 = fig.add_subplot(122)
        y_pos = np.arange(len(config.CLASES))
        ax2.barh(y_pos, probabilidades * 100)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(config.CLASES)
        ax2.set_xlabel('Probabilidad (%)')
        ax2.set_title('Probabilidades por clase')
        
        fig.tight_layout()
        
        # Mostrar figura en canvas
        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar resultados de texto
        results_text = f"Resultado: {config.CLASES[clase_id]}\n\n"
        for i, clase in enumerate(config.CLASES):
            results_text += f"{clase}: {probabilidades[i]*100:.2f}%\n"
        
        tk.Label(self.result_frame, text=results_text, justify=tk.LEFT, font=("Arial", 12)).pack(pady=10)
        
        # Actualizar estado
        self.lbl_status.config(text=f"Predicción completada: {config.CLASES[clase_id]}")

def main():
    # Imprimir información de depuración
    print("Iniciando aplicación...")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Verificar configuración
    print(f"Directorio de modelos: {config.MODELS_DIR}")
    print(f"Modelo a cargar: {config.MEJOR_MODELO}")
    print(f"Clases detectadas: {config.CLASES}")
    
    # Iniciar aplicación
    root = tk.Tk()
    app = SimpleButtonClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()