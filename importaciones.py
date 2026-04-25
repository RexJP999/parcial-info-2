import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import loadmat, whosmat
import os

Carpeta = "Imagenes"
os.makedirs(Carpeta, exist_ok=True)

def listar_archivos(carpeta):
    ruta = os.path.join("data", carpeta)

    if not os.path.exists(ruta):
        print("La carpeta no existe")
        return []

    archivos = [f for f in os.listdir(ruta) if f.endswith(".mat")]
    return archivos

def seleccionar_archivo(carpeta):
    archivos = listar_archivos(carpeta)

    if not archivos:
        print("No hay archivos disponibles")
        return None

    print(f"\nArchivos en {carpeta}:")

    for i, f in enumerate(archivos):
        codigo = f.split("_")[0] 
        print(f"{i+1}. {codigo}")

    try:
        op = int(input("Seleccione archivo: ")) - 1
        return os.path.join("data", carpeta, archivos[op])
    except:
        print("Opción inválida")
        return None
    
class ArchivoEEG:
    def __init__(self,ruta):
        self.__ruta = ruta
        self.__data = None
        self.__matriz = None
        self.__key = None
       

    def cargar_archivo(self):
        self.__data = loadmat(self.__ruta)

    def mostrar_llaves(self):
        print(whosmat(self.__ruta))

    def set_key(self, key):
        if key not in self.__data:
            raise ValueError("No se encontro la variable en el .mat")
        self.__key = key
        self.__matriz = self.__data[key]

    def get_keys(self):
        if self.__data is None:
            raise ValueError("Primero cargue el archivo")
        return [k for k in self.__data.keys() if not k.startswith("__")]

    def __obtener_matriz_2D(self):
        if self.__matriz is None:
            raise ValueError("No hay matriz cargada")

        matriz = np.squeeze(self.__matriz)
        shape = matriz.shape

        if matriz.ndim == 3:
            canales, muestras, ensayos = shape
            duracion = muestras/ 1000

            print("La matriz tiene:")
            print(f"Canales: {canales}")
            print(f"Muestras (tiempo): {muestras} -> Duracion: {duracion:.2f} s")
            print(f"Ensayos: {ensayos}")
            matriz = np.mean(matriz, axis=2)

        elif matriz.ndim == 2:
            canales, muestras = shape
            duracion = muestras / 1000

            print(f"Canales: {canales}")
            print(f"Muestras (tiempo): {muestras} → Duración: {duracion:.2f} s")
            print("Ensayos: 1")
        
        else:
            raise ValueError("Dimension no soportada")

        print("Shape para análisis:", matriz.shape)

        return matriz
    
    def sumar_canales(self, canales, inicio, fin):

        matriz = self.__obtener_matriz_2D()

        if min(canales) < 0:
            raise ValueError("Canales no pueden ser negativos")
        
        if len(canales) != 3:
            raise ValueError("Debe seleccionar exactamente 3 canales")

        if max(canales) >= matriz.shape[0]:
            raise ValueError("Canal fuera de rango")

        if inicio < 0 or fin > matriz.shape[1] or inicio >= fin:
            raise ValueError("Rango inválido")

        tiempo = np.arange(inicio, fin) / 1000

        suma = np.sum(matriz[canales, inicio:fin], axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        for c in canales:
            axs[0].plot(tiempo, matriz[c, inicio:fin], label=f"Canal {c}")

        axs[0].set_title("Canales EEG")
        axs[0].set_xlabel("Tiempo (s)")
        axs[0].set_ylabel("Amplitud (µV)")
        axs[0].legend()

        axs[1].plot(tiempo, suma)
        axs[1].set_title("Suma de canales")
        axs[1].set_xlabel("Tiempo (s)")
        axs[1].set_ylabel("Amplitud (µV)")

        nombre = f"eeg_suma_{self.__key}.png"

        plt.tight_layout()
        plt.savefig(os.path.join("Imagenes", nombre))
        plt.show()

    def estadisticas(self):
        if self.__matriz is None:
            raise ValueError("No hay ninguna matriz cargada")
        
        matriz = np.squeeze(self.__matriz)
        print("Forma orignal de la matriz:", matriz.shape)

        if matriz.ndim == 3:
            # La matriz tiene forma (canales, muestras, ensayos)
            # Se calcula el promedio y desviación estándar sobre los ensayos (axis=2)
            promedio = np.mean(matriz, axis=2)
            std = np.std(matriz, axis=2)
            # Posteriormente promediamos entre canales para obtener una señal global
            promedio_plot = np.mean(promedio, axis = 0)
            std_plot = np.mean(std, axis = 0)

        elif matriz.ndim == 2:
            promedio_plot = np.mean(matriz, axis=0)
            std_plot = np.std(matriz, axis=0)

        else:
            raise ValueError("Dimension no soportada")
        
        fig, axs = plt.subplots(2,1, figsize=(10, 8))
        axs[0].stem(promedio_plot) 
        axs[0].set_title("Promedio")
        axs[0].set_xlabel("Muestras")
        axs[0].set_ylabel("Amplitud (µV)")
        axs[0].legend(["Promedio"])

        axs[1].stem(std_plot)
        axs[1].set_title("Desviación estándar")
        axs[1].set_xlabel("Muestras")
        axs[1].set_ylabel("Amplitud (µV)")
        axs[1].legend(["Desviacion estandar"])

        plt.tight_layout()
        nombre = f"eeg_stats_{self.__key}.png"
        plt.savefig(os.path.join("Imagenes", nombre))
        plt.show()
        

