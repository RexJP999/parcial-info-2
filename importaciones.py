import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, whosmat
import os
from datetime import datetime
import hashlib

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

class ArchivoSIATA:
    def __init__(self, ruta_archivo):
        #Constructor: Inicializa el objeto con la ruta del archivo CSV
        self.ruta = ruta_archivo
        self.nombre_archivo = os.path.basename(ruta_archivo)
        self.tipo = "SIATA"
        self.fecha_carga = datetime.now()
        self.datos = None
        self.cargar_datos()

    def get_info_resumida(self):
        #Devuelve información resumida del objeto
        return {
            'id': self.id,
            'nombre': self.nombre_archivo,
            'tipo': self.tipo,
            'fecha_carga': self.fecha_carga.strftime("%Y-%m-%d %H:%M:%S"),
            'filas': self.datos.shape[0],
            'columnas': self.datos.shape[1],
            'ruta': self.ruta
        }

    def cargar_datos(self):
        #Carga el archivo CSV usando pandas. Detecta automáticamente la primera columna como fechas
        try:
            self.datos = pd.read_csv(self.ruta, parse_dates=[0], infer_datetime_format=True)

            print(f"✓ Archivo {self.nombre_archivo} cargado exitosamente")
            print(f"  Dimensiones: {self.datos.shape[0]} filas x {self.datos.shape[1]} columnas")
            print(f"  Rango de fechas: {self.datos.iloc[:,0].min()} a {self.datos.iloc[:,0].max()}")

            # Identificar columnas numéricas y de fecha
            print(f"\n Columnas encontradas en {self.nombre_archivo}:")
            for i, col in enumerate(self.datos.columns, 1):
                tipo = "Fecha" if "fecha" in col.lower() or col == self.datos.columns[0] else "Numérica" if pd.api.types.is_numeric_dtype(self.datos[col]) else "Texto"
                print(f"  {i}. {col} - {tipo}")

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.ruta}")
            raise
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            print("  Intentando carga sin parseo de fechas...")
            try:
                self.datos = pd.read_csv(self.ruta)
                print("Archivo cargado")
            except Exception as e2:
                print(f"Error también en carga simple: {e2}")
                raise


    def mostrar_info_basica(self):
        #Muestra información básica del DataFrame usando info y describe

        print("\n" + "="*70)
        print(f"INFORMACIÓN DEL ARCHIVO: {self.nombre_archivo}")
        print("="*70)

        print("\n--- INFO() ---")
        print("Información general del DataFrame:")

        # Capturar info en str para mostrarlo mejor
        import io
        buffer = io.StringIO()
        self.datos.info(buf=buffer)
        info_str = buffer.getvalue()
        print(info_str)

        print("\n--- DESCRIBE() ---")
        print("Estadísticas descriptivas de columnas numéricas:")
        print(self.datos.describe())

        print("\n--- PRIMERAS 10 FILAS ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.datos.head(10))

        print("\n--- ÚLTIMAS 10 FILAS ---")
        print(self.datos.tail(10))

        # Mostrar estadísticas adicionales
        print("\n--- ESTADÍSTICAS ADICIONALES ---")
        print(f"Total de registros: {len(self.datos)}")
        print(f"Valores nulos por columna:")
        print(self.datos.isnull().sum())

    def listar_columnas_numericas(self):
        #Lista las columnas numéricas disponibles para graficar
        columnas_numericas = self.datos.select_dtypes(include=[np.number]).columns.tolist()

        if not columnas_numericas:
            print("No se encontraron columnas numéricas en el archivo")
            return []

        print("\n Columnas numéricas disponibles:")
        for i, col in enumerate(columnas_numericas, 1):
            # Mostrar estadísticas básicas de cada columna
            media = self.datos[col].mean()
            std = self.datos[col].std()
            print(f"  {i}. {col} - Media: {media:.2f}, Std: {std:.2f}")

        return columnas_numericas


    def graficar_tres_tipos(self, columna):
        #Crea 3 subplots: plot, boxplot e histograma de una columna específica
        if columna not in self.datos.columns:
            print(f"La columna '{columna}' no existe en el archivo")
            columnas_num = self.listar_columnas_numericas()
            if columnas_num:
                print(f"  Columnas numéricas disponibles: {columnas_num}")
            return None

        # Verificar que la columna sea numérica
        if not pd.api.types.is_numeric_dtype(self.datos[columna]):
            print(f"La columna '{columna}' no es numérica")
            return None

        # Eliminar valores nulos para mejor visualización
        datos_limpios = self.datos[columna].dropna()

        if len(datos_limpios) == 0:
            print(f"✗ La columna '{columna}' no tiene datos válidos")
            return None

        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Análisis de: {columna}\n{self.nombre_archivo}', fontsize=14, fontweight='bold')

        # Subplot 1: Gráfico de líneas (plot) - usando índice o fecha si está disponible
        if isinstance(self.datos.index, pd.DatetimeIndex) or self.datos.columns[0] in ['fecha', 'date', 'datetime']:
            # Si tenemos fechas, usar para el eje X
            if isinstance(self.datos.index, pd.DatetimeIndex):
                x_values = self.datos.index
            else:
                # Intentar usar la primera columna como fechas
                fecha_col = self.datos.columns[0]
                x_values = pd.to_datetime(self.datos[fecha_col])
            axes[0].plot(x_values, datos_limpios, color='blue', alpha=0.7, linewidth=0.5)
            axes[0].set_xlabel('Fecha')
        else:
            axes[0].plot(range(len(datos_limpios)), datos_limpios, color='blue', alpha=0.7)
            axes[0].set_xlabel('Índice')

        axes[0].set_title('Serie Temporal')
        axes[0].set_ylabel(columna)
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Boxplot
        axes[1].boxplot(datos_limpios, vert=True)
        axes[1].set_title('Diagrama de Caja')
        axes[1].set_ylabel(columna)
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: Histograma
        axes[2].hist(datos_limpios, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[2].set_title('Histograma')
        axes[2].set_xlabel(columna)
        axes[2].set_ylabel('Frecuencia')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def aplicar_operaciones(self, columna1, columna2, operacion):

        if columna1 not in self.datos.columns:
            print(f"✗ La columna '{columna1}' no existe")
            return None

        if not pd.api.types.is_numeric_dtype(self.datos[columna1]):
            print(f"✗ La columna '{columna1}' no es numérica")
            return None

        if operacion in ['suma', 'resta']:
            if columna2 not in self.datos.columns:
                print(f"✗ La columna '{columna2}' no existe")
                return None
            if not pd.api.types.is_numeric_dtype(self.datos[columna2]):
                print(f"✗ La columna '{columna2}' no es numérica")
                return None

    # ================= APPLY =================
        if operacion == 'apply':
            serie = self.datos[columna1]

            min_val = serie.min()
            max_val = serie.max()

            resultado = serie.apply(
                lambda x: (x - min_val) / (max_val - min_val)
                if pd.notna(x) and max_val != min_val else np.nan
            )

            print(f"\n✓ APPLY - Normalización de '{columna1}'")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            fig.suptitle(f'Normalización de {columna1}', fontsize=14)

            ax1.plot(serie.values)
            ax1.set_title('Datos Originales')

            ax2.plot(resultado.values)
            ax2.set_title('Datos Normalizados')

    # ================= MAP =================
        elif operacion == 'map':
            serie = self.datos[columna1]

            p25 = serie.quantile(0.25)
            p75 = serie.quantile(0.75)

            def clasificar(x):
                if pd.isna(x):
                    return 'Sin dato'
                elif x <= p25:
                    return 'Bajo'
                elif x <= p75:
                    return 'Medio'
                else:
                    return 'Alto'

            resultado = serie.map(clasificar)

            print(f"\n✓ MAP - Clasificación de '{columna1}'")

            fig, ax = plt.subplots(figsize=(10, 6))
            resultado.value_counts().plot(kind='bar', ax=ax)

    # ================= SUMA =================
        elif operacion == 'suma':
            serie1 = self.datos[columna1]
            serie2 = self.datos[columna2]

            resultado = serie1 + serie2

            print(f"\n✓ SUMA: {columna1} + {columna2}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

            ax1.plot(serie1.values, label=columna1)
            ax1.plot(serie2.values, label=columna2)
            ax1.legend()

            ax2.plot(resultado.values)
            ax2.set_title("Resultado suma")

    # ================= RESTA =================
        elif operacion == 'resta':
            serie1 = self.datos[columna1]
            serie2 = self.datos[columna2]

            resultado = serie1 - serie2

            print(f"\n✓ RESTA: {columna1} - {columna2}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

            ax1.plot(serie1.values, label=columna1)
            ax1.plot(serie2.values, label=columna2)
            ax1.legend()

            ax2.plot(resultado.values)
            ax2.set_title("Resultado resta")

        else:
            print("Operación no válida")
            return None

    # ================= GUARDAR =================
        nombre_imagen = f"{self.nombre_archivo.replace('.csv', '')}_{operacion}.png"

        fig.savefig(nombre_imagen, dpi=100, bbox_inches='tight')
        print(f"\n✓ Gráfico guardado como: {nombre_imagen}")

        plt.show()

        return resultado



    def convertir_fecha_a_indice(self):
        #Convierte la primera columna (fecha) a índice del DataFrame. Detecta automáticamente la columna de fechas
        # Buscar columna de fecha (normalmente la primera)
        posibles_fechas = [col for col in self.datos.columns if 'fecha' in col.lower() or 'date' in col.lower()]

        if posibles_fechas:
            columna_fecha = posibles_fechas[0]
        else:
            # Asumir que la primera columna es la fecha
            columna_fecha = self.datos.columns[0]

        try:
            # Convertir la columna a datetime
            self.datos[columna_fecha] = pd.to_datetime(self.datos[columna_fecha])
            # Establecer como índice
            self.datos.set_index(columna_fecha, inplace=True)
            print(f"\n Columna '{columna_fecha}' convertida a índice correctamente")
            print(f"  Rango de fechas: {self.datos.index.min()} a {self.datos.index.max()}")
            print(f"  Total de días: {(self.datos.index.max() - self.datos.index.min()).days}")
            return True
        except Exception as e:
            print(f" Error al convertir fecha: {e}")
            return False

    def remuestrear_y_graficar(self):
        #Realiza remuestreo a días, meses y trimestral y los grafica
        # Verificar que el índice sea datetime
        if not isinstance(self.datos.index, pd.DatetimeIndex):
            print("\n El índice no es de tipo fecha. Convirtiendo...")
            if not self.convertir_fecha_a_indice():
                print("✗ No se pudo convertir a índice de fechas")
                return None

        # Seleccionar columnas numéricas
        columnas_numericas = self.datos.select_dtypes(include=[np.number]).columns

        if len(columnas_numericas) == 0:
            print("✗ No hay columnas numéricas para remuestrear")
            return None

        # Mostrar opciones al usuario
        print("\nColumnas numéricas disponibles para remuestrear:")
        for i, col in enumerate(columnas_numericas, 1):
            print(f"  {i}. {col}")

        try:
            opcion = int(input("\nSeleccione una columna (número): ")) - 1
            if 0 <= opcion < len(columnas_numericas):
                columna = columnas_numericas[opcion]
            else:
                columna = columnas_numericas[0]
                print(f"  Usando columna por defecto: {columna}")
        except:
            columna = columnas_numericas[0]
            print(f"  Usando columna por defecto: {columna}")

        print(f"\nRemuestreando columna: {columna}")

        # Realizar remuestreos con diferentes métodos
        diario_mean = self.datos[columna].resample('D').mean()
        diario_median = self.datos[columna].resample('D').median()

        mensual_mean = self.datos[columna].resample('M').mean()
        mensual_median = self.datos[columna].resample('M').median()

        trimestral_mean = self.datos[columna].resample('Q').mean()
        trimestral_median = self.datos[columna].resample('Q').median()

        # Crear gráficos
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Remuestreo Temporal - {columna}\n{self.nombre_archivo}', fontsize=14, fontweight='bold')

        # Diario (2 subplots)
        axes[0, 0].plot(diario_mean.index, diario_mean.values, color='blue', linewidth=1)
        axes[0, 0].set_title(' Diario - Media')
        axes[0, 0].set_ylabel(columna)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(diario_median.index, diario_median.values, color='cyan', linewidth=1)
        axes[0, 1].set_title(' Diario - Mediana')
        axes[0, 1].grid(True, alpha=0.3)

        # Mensual
        axes[1, 0].plot(mensual_mean.index, mensual_mean.values, color='green', marker='o', markersize=4, linewidth=1.5)
        axes[1, 0].set_title(' Mensual - Media')
        axes[1, 0].set_ylabel(columna)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(mensual_median.index, mensual_median.values, color='lightgreen', marker='s', markersize=4, linewidth=1.5)
        axes[1, 1].set_title('Mensual - Mediana')
        axes[1, 1].grid(True, alpha=0.3)

        # Trimestral
        axes[2, 0].plot(trimestral_mean.index, trimestral_mean.values, color='red', marker='^', markersize=6, linewidth=2)
        axes[2, 0].set_title('Trimestral - Media')
        axes[2, 0].set_xlabel('Fecha')
        axes[2, 0].set_ylabel(columna)
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(trimestral_median.index, trimestral_median.values, color='orange', marker='v', markersize=6, linewidth=2)
        axes[2, 1].set_title(' Trimestral - Mediana')
        axes[2, 1].set_xlabel('Fecha')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Guardar gráfico
        nombre_archivo = f"{self.nombre_archivo.replace('.csv', '')}_remuestreo.png"
        fig.savefig(nombre_archivo, dpi=100, bbox_inches='tight')
        print(f"\n Gráfico de remuestreo guardado como: {nombre_archivo}")

        # Mostrar estadísticas del remuestreo
        print("\nEstadísticas del remuestreo:")
        print(f"  Datos originales: {len(self.datos[columna].dropna())} puntos")
        print(f"  Datos diarios: {len(diario_mean.dropna())} puntos")
        print(f"  Datos mensuales: {len(mensual_mean.dropna())} puntos")
        print(f"  Datos trimestrales: {len(trimestral_mean.dropna())} puntos")

        return fig

    def analisis_correlacion(self):
        #Análisis de correlación entre variables ambientales
        # Seleccionar solo columnas numéricas
        columnas_numericas = self.datos.select_dtypes(include=[np.number]).columns

        if len(columnas_numericas) < 2:
            print("Se necesitan al menos 2 columnas numéricas para correlación")
            return None

        print("\n" + "="*70)
        print("MATRIZ DE CORRELACIÓN")
        print("="*70)

        # Calcular matriz de correlación
        corr_matrix = self.datos[columnas_numericas].corr()
        print(corr_matrix)

        # Graficar heatmap de correlación
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Configurar ejes
        ax.set_xticks(range(len(columnas_numericas)))
        ax.set_yticks(range(len(columnas_numericas)))
        ax.set_xticklabels(columnas_numericas, rotation=45, ha='right')
        ax.set_yticklabels(columnas_numericas)

        # Añadir valores numéricos en cada celda
        for i in range(len(columnas_numericas)):
            for j in range(len(columnas_numericas)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                             fontsize=10)

        ax.set_title(f'Matriz de Correlación - {self.nombre_archivo}', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Coeficiente de Correlación')
        plt.tight_layout()

        # Guardar gráfico
        nombre_archivo = f"{self.nombre_archivo.replace('.csv', '')}_correlacion.png"
        fig.savefig(nombre_archivo, dpi=100, bbox_inches='tight')
        print(f"\nHeatmap de correlación guardado como: {nombre_archivo}")

        # Mostrar correlaciones más fuertes
        print("\n Correlaciones más fuertes (|r| > 0.7):")
        for i in range(len(columnas_numericas)):
            for j in range(i+1, len(columnas_numericas)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    print(f"  {columnas_numericas[i]} corr {columnas_numericas[j]}: {corr_val:.3f}")

        return corr_matrix

class ArchivoEEG:
    def __init__(self,ruta):
        self.__ruta = ruta
        self.__data = None
        self.__matriz = None
        self.__key = None
        self.id = os.path.basename(ruta)
        self.nombre_archivo = os.path.basename(ruta)
        self.tipo = "EEG"
        self.ruta = ruta
        self.fecha_carga = datetime.now()


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

        nombre = f"eeg_suma_{self.__key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

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
        nombre = f"eeg_stats_{self.__key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(os.path.join("Imagenes", nombre))
        plt.show()

class GestorArchivos:

    def __init__(self):
        self.objetos = {}

    def agregar_objeto(self, objeto):
        self.objetos[objeto.id] = objeto
        print(f"Objeto guardado: {objeto.id}")

    def listar_todos(self):
        if not self.objetos:
            print("No hay objetos guardados")
            return

        print("\nObjetos guardados:")
        for obj in self.objetos.values():
            print(f"- {obj.id} ({obj.tipo})")

    def buscar_por_nombre(self, nombre):
        encontrados = []
        for obj in self.objetos.values():
            if nombre.lower() in obj.nombre_archivo.lower():
                encontrados.append(obj)
    
    def buscar_por_tipo(self, tipo):
        resultados = []

        for obj in self.objetos.values():
            if obj.tipo == tipo:
                resultados.append(obj)

        return resultados

        if encontrados:
            for obj in encontrados:
                print(f"- {obj.id} ({obj.tipo})")
        else:
            print("No se encontraron resultados")