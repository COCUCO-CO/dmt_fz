#!/usr/bin/env python3
"""
Script para ejecutar el procesamiento completo de archivos EEG

Este script:
1. Lee archivos .set de EEGLAB (preprocesados con ICA)
2. Calcula forward/inverse solution
3. Extrae fases con Hilbert
4. Calcula sincronización y parámetro de Kuramoto
5. Guarda resultados en archivos pickle

Uso:
    cd /home/tomas/PycharmProjects/dmt_fz/dmt
    python run_fwd_processing.py
"""

import os
import sys

# Asegurarse de que estamos en el directorio correcto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Importar el módulo fwd
print("="*60)
print("Inicializando procesamiento de EEG")
print("="*60)

# Ejecutar las celdas de configuración de fwd.py
exec(open('fwd.py').read(), globals())

#%%

def run_processing():
    """
    Ejecuta el procesamiento completo de todos los archivos
    """
    print("\n" + "="*60)
    print("INICIANDO PROCESAMIENTO DE TODOS LOS ARCHIVOS")
    print("="*60)
    
    total_files = len(dmt_files_list) + len(eyes_closed_files_list) + len(eyes_open_files_list)
    current_file = 0
    
    # Procesar archivos DMT
    print(f"\n{'#'*60}")
    print(f"### PROCESANDO CONDICIÓN: DMT ({len(dmt_files_list)} archivos)")
    print(f"{'#'*60}")
    
    for file in dmt_files_list:
        current_file += 1
        print(f"\n[{current_file}/{total_files}] Archivo: {os.path.basename(file)}")
        try:
            do_the_math(file, condition="DMT")
        except Exception as e:
            print(f"❌ ERROR procesando {file}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Procesar archivos EC (Eyes Closed)
    print(f"\n{'#'*60}")
    print(f"### PROCESANDO CONDICIÓN: EC ({len(eyes_closed_files_list)} archivos)")
    print(f"{'#'*60}")
    
    for file in eyes_closed_files_list:
        current_file += 1
        print(f"\n[{current_file}/{total_files}] Archivo: {os.path.basename(file)}")
        try:
            do_the_math(file, condition="EC")
        except Exception as e:
            print(f"❌ ERROR procesando {file}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Procesar archivos EO (Eyes Open)
    print(f"\n{'#'*60}")
    print(f"### PROCESANDO CONDICIÓN: EO ({len(eyes_open_files_list)} archivos)")
    print(f"{'#'*60}")
    
    for file in eyes_open_files_list:
        current_file += 1
        print(f"\n[{current_file}/{total_files}] Archivo: {os.path.basename(file)}")
        try:
            do_the_math(file, condition="EO")
        except Exception as e:
            print(f"❌ ERROR procesando {file}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print("\n" + "="*60)
    print("PROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"✓ Total de archivos procesados: {current_file}")
    print(f"✓ DMT: {len(dmt_files_list)} archivos")
    print(f"✓ EC:  {len(eyes_closed_files_list)} archivos")
    print(f"✓ EO:  {len(eyes_open_files_list)} archivos")
    print(f"\nResultados guardados en:")
    print(f"  {output_folder}")
    print("="*60)


if __name__ == "__main__":
    run_processing()

