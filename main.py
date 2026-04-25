import sys
import os
import glob
from importaciones import *
import matplotlib.pyplot as plt

gestor = GestorArchivos()

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_archivos_siata_disponibles():
    #Muestra los archivos CSV disponibles
    archivos = glob.glob("CalAir_VA_*.csv")
    if archivos:
        print("\nArchivos CalAir_VA encontrados:")
        for i, archivo in enumerate(sorted(archivos), 1):
            tamaño = os.path.getsize(archivo) / 1024
            print(f"  {i}. {os.path.basename(archivo)} ({tamaño:.0f} KB)")
        return archivos
    else:
        print("\nNo se encontraron archivos CalAir_VA_*.csv")
        return []


def mostrar_menu_principal():
    #Muestra el menú principal
    print("\n" + "="*60)
    print("SISTEMA DE ANÁLISIS DE DATOS")
    print("Neurología - Calidad del aire y Enfermedades Neurodegenerativas")
    print("="*60)
    print("\nOPCIONES:")
    print("  1. Procesar archivo SIATA (CSV - Calidad del Aire)")
    print("  2. Procesar archivo EEG (MAT - Señales Cerebrales)")
    print("  3. Gestionar archivo")
    print("  4. Salir")
    print("-"*60)

def menu_siata():
    #Menú para procesar archivos SIATA
    archivos_disponibles = mostrar_archivos_siata_disponibles()

    if archivos_disponibles:
        print("\n0. Ingresar ruta manualmente")
        try:
            opcion = int(input("\nSeleccione una opción: "))
            if opcion == 0:
                ruta = input("Ingrese la ruta del archivo CSV: ")
            elif 1 <= opcion <= len(archivos_disponibles):
                ruta = sorted(archivos_disponibles)[opcion - 1]
            else:
                print("Opción inválida")
                return
        except ValueError:
            ruta = input("Ingrese la ruta del archivo CSV: ")
    else:
        ruta = input("Ingrese la ruta del archivo CSV: ")

    try:
        siata = ArchivoSIATA(ruta)

        while True:
            print("\n" + "-"*50)
            print(f" SUBMENÚ SIATA - {siata.nombre_archivo}")
            print("-"*50)
            print("1. Mostrar información básica (info y describe)")
            print("2. Graficar columna (plot, boxplot, histograma)")
            print("3. Aplicar operaciones (apply, map, suma/resta)")
            print("4. Convertir fecha a índice y remuestrear")
            print("5. Análisis de correlación entre variables")
            print("6. Volver al menú principal")
            print("-"*50)

            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                siata.mostrar_info_basica()
                input("\nPresione Enter para continuar...")

            elif opcion == '2':
                columnas_num = siata.listar_columnas_numericas()
                if not columnas_num:
                    print(" No hay columnas numéricas")
                else:
                    print("\nColumnas numéricas disponibles:")
                    for i, col in enumerate(columnas_num, 1):
                        print(f"  {i}. {col}")
                    columna = input("¿Qué columna desea graficar? (nombre o número): ")
                    if columna.isdigit():
                        idx = int(columna) - 1
                        if 0 <= idx < len(columnas_num):
                            columna = columnas_num[idx]
                    fig = siata.graficar_tres_tipos(columna)
                    if fig:
                        plt.show()
                        nombre = input("Nombre para guardar (Enter para usar default): ")
                        if nombre:
                            fig.savefig(f"{nombre}.png", dpi=100, bbox_inches='tight')
                input("\nPresione Enter para continuar...")

            elif opcion == '3':
                columnas_num = siata.listar_columnas_numericas()
                

                if not columnas_num:
                        return

                # Mostrar columnas numeradas
                print("\nSeleccione columnas por número:")
                for i, col in enumerate(columnas_num, 1):
                    print(f"{i}. {col}")

                try:
                    i1 = int(input("Seleccione número de la primera columna: ")) - 1
                    i2 = int(input("Seleccione número de la segunda columna: ")) - 1

                    col1 = columnas_num[i1]
                    col2 = columnas_num[i2]

                except:
                    print("Selección inválida")
                    input("\nPresione Enter para continuar...")
                    return

                print("\nOperaciones:")
                print("1. apply (normalización)")
                print("2. map (clasificación)")
                print("3. suma")
                print("4. resta")
                op_oper = input("Opción: ")

                operacion_map = {'1': 'apply', '2': 'map', '3': 'suma', '4': 'resta'}
                if op_oper in operacion_map:
                    siata.aplicar_operaciones(col1, col2, operacion_map[op_oper])
                    plt.show()
                input("\nPresione Enter para continuar...")

            elif opcion == '4':
                siata.remuestrear_y_graficar()
                plt.show()
                input("\nPresione Enter para continuar...")

            elif opcion == '5':
                siata.analisis_correlacion()
                plt.show()
                input("\nPresione Enter para continuar...")

            elif opcion == '6':
                break
            else:
                print("Opción inválida")

    except Exception as e:
        print(f"✗ Error: {e}")
        input("\nPresione Enter para continuar...")

def menu_gestion():
    while True:
        print("\n--- GESTIÓN ---")
        print("1. Listar objetos")
        print("2. Buscar por nombre")
        print("3. Salir")

        op = input("Opción: ")

        if op == '1':
            gestor.listar_todos()

        elif op == '2':
            nombre = input("Ingrese nombre: ")
            gestor.buscar_por_nombre(nombre)

        elif op == '3':
            break

        else:
            print("Opción inválida")


def menu_eeg():
    print("\n" + "-"*50)
    print(" MENÚ EEG - Señales Cerebrales (.mat)")
    print("-"*50)

    # Selección tipo de archivo
    print("1. Control")
    print("2. Parkinson")
    print("0. Ingresar ruta manualmente")

    opcion = input("Seleccione una opción: ")

    if opcion == '1':
        carpeta = "control"
    elif opcion == '2':
        carpeta = "parkinson"
    elif opcion == '0':
        ruta = input("Ingrese la ruta del archivo .mat: ")
    else:
        print("Opción inválida")
        return

    # Selección automática desde carpeta
    if opcion in ['1', '2']:
        ruta_base = carpeta

        if not os.path.exists(ruta_base):
            print("La carpeta no existe")
            return

        archivos = [f for f in os.listdir(ruta_base) if f.endswith(".mat")]

        if not archivos:
            print("No hay archivos disponibles")
            return

        print(f"\nArchivos en {carpeta}:")
        for i, f in enumerate(sorted(archivos), 1):
            codigo = f.split("_")[0]
            print(f"{i}. {codigo}")

        try:
            op = int(input("Seleccione archivo: ")) - 1
            ruta = os.path.join(ruta_base, archivos[op])
        except:
            print("Opción inválida")
            return

    # Crear objeto EEG
    try:
        eeg = ArchivoEEG(ruta)
        eeg.cargar_archivo()

        gestor.agregar_objeto(eeg)

        print("\nLlaves disponibles en el archivo:")
        eeg.mostrar_llaves()

        # Selección de variable
        keys = eeg.get_keys()
        print("\nVariables disponibles:")
        for i, k in enumerate(keys, 1):
            print(f"{i}. {k}")

        op_key = int(input("Seleccione variable: ")) - 1
        key = keys[op_key]

        eeg.set_key(key)

        # Submenú EEG
        while True:
            print("\n" + "-"*50)
            print(f" SUBMENÚ EEG - {os.path.basename(ruta)}")
            print("-"*50)
            print("1. Sumar 3 canales en un intervalo")
            print("2. Promedio y desviación estándar (stem)")
            print("3. Volver al menú principal")
            print("-"*50)

            opcion_sub = input("Seleccione una opción: ")

            if opcion_sub == '1':
                try:
                    canales = list(map(int, input("Ingrese 3 canales (ej: 0 1 2): ").split()))
                    inicio = int(input("Inicio (muestra): "))
                    fin = int(input("Fin (muestra): "))

                    eeg.sumar_canales(canales, inicio, fin)

                except Exception as e:
                    print(f"Error: {e}")

                input("\nPresione Enter para continuar...")

            elif opcion_sub == '2':
                try:
                    eeg.estadisticas()
                except Exception as e:
                    print(f"Error: {e}")

                input("\nPresione Enter para continuar...")

            elif opcion_sub == '3':
                break

            else:
                print("Opción inválida")

    except Exception as e:
        print(f"❌Error: {e}")
        input("\nPresione Enter para continuar...")

def main():
    print("\n" + "="*60)
    print("INICIANDO SISTEMA...")
    print("="*60)

    while True:
        mostrar_menu_principal()
        opcion = input("\nSeleccione una opción (1-4): ")

        if opcion == '1':
            menu_siata()
        elif opcion == '2':
            menu_eeg()
        elif opcion == '3':
            menu_gestion()
        elif opcion == "4":
            print("Gracias por usar el sistema hasta luego")
            break

        else:
            print("\n✗ Opción inválida")
            input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()