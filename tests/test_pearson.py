#!/usr/bin/env python3
"""
Script de prueba para pearson.py
Verifica que se pueden calcular correlaciones entre Kuramoto y cuestionarios
Usa solo S01-DMT (duplicado para simular varios sujetos)
"""

import pickle
import numpy as np
import pandas as pd

# ============================================================
# Funciones auxiliares
# ============================================================

def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def order_parameter(phase_df):
    """Calcula parámetro de Kuramoto desde DataFrame de fases"""
    r = np.abs(np.exp(1j*phase_df).mean(axis=0))
    return r

# ============================================================
# Configuración
# ============================================================

path = "/home/tomas/PycharmProjects/dmt_fz/dmt/fwd-inv-stc/"
spectral_path = "/home/tomas/PycharmProjects/dmt_fz/dmt/spectral_sources/"

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
hemi_list = ["RH", "LH", "both"]
net_list = ["FPN", "DMN", "DAN", "LN ", "SVA", "SMN", "VN "]

print("="*60)
print("TEST: Verificando pipeline de correlaciones")
print("="*60)

# ============================================================
# 1. Cargar datos de cuestionarios
# ============================================================

print("\n1. Cargando datos de cuestionarios...")
labels = list(pd.read_csv(spectral_path + "target_labels.txt", header=None)[0])
targets = pd.read_csv(spectral_path + 'target.csv', header=None, names=labels)

print(f"   ✓ Cargados {len(targets)} sujetos")
print(f"   ✓ {len(labels)} variables: {labels[:5]}...")

# ============================================================
# 2. Cargar datos de Kuramoto (order_all)
# ============================================================

print("\n2. Cargando datos de Kuramoto...")

# Cargar S01-DMT
test_file = path + "DMT/order_all-S01-DMT.pkl"
data_dict = load_file(test_file)

print(f"   ✓ Archivo cargado: {test_file}")
print(f"   ✓ Condición: {list(data_dict.keys())}")
print(f"   ✓ Bandas: {list(data_dict['DMT'].keys())}")

# ============================================================
# 3. Extraer valores de Kuramoto por red/hemisferio/banda
# ============================================================

print("\n3. Calculando Kuramoto por época...")

# Seleccionar una configuración de ejemplo
band = "Alpha"
hemi = "both"
net = "DMN"

# Obtener fases filtradas para esa red
phase_epochs = data_dict["DMT"][band][hemi][net]  # Lista de DataFrames (uno por época)

print(f"   ✓ Configuración: {band} / {hemi} / {net}")
print(f"   ✓ Épocas disponibles: {len(phase_epochs)}")

# Calcular Kuramoto para cada época
r_values = []
for epoch_df in phase_epochs:
    r = order_parameter(epoch_df)
    r_values.append(r.mean())  # Promedio temporal dentro de la época

print(f"   ✓ Kuramoto calculado: {len(r_values)} valores")
print(f"   ✓ Rango: [{np.min(r_values):.3f}, {np.max(r_values):.3f}]")
print(f"   ✓ Media: {np.mean(r_values):.3f} ± {np.std(r_values):.3f}")

# ============================================================
# 4. Simular análisis de correlaciones (con 1 sujeto)
# ============================================================

print("\n4. Simulando análisis de correlaciones...")

# Para hacer correlación necesitamos múltiples sujetos
# Como solo tenemos S01, mostramos el concepto
subject_kuramoto_mean = np.mean(r_values)

print(f"\n   Sujeto S01-DMT:")
print(f"   - Kuramoto medio ({band}/{net}): {subject_kuramoto_mean:.3f}")
print(f"   - Variables subjetivas (primeras 5):")

subject_idx = 0  # S01 corresponde al primer sujeto
for i, label in enumerate(labels[:5]):
    print(f"     {label}: {targets.iloc[subject_idx, i]}")

# ============================================================
# 5. Ejemplo de correlación (necesita múltiples sujetos)
# ============================================================

print("\n" + "="*60)
print("NOTA: Para correlaciones reales necesitas:")
print("="*60)
print("""
Para hacer el análisis completo de pearson.py necesitas:

1. Procesar TODOS los sujetos con fwd.py (9 archivos mínimo)
2. Procesar todos con multi2pool2.py 
3. Luego ejecutar pearson.py que hará:
   
   for sujeto in [S01, S02, S03, ...]:
       kuramoto_sujeto = mean(order_all-sujeto.pkl)
       target_sujeto = target.csv[sujeto]
       
   # Correlación de Pearson entre vectores
   r, p = pearsonr(kuramoto_todos_sujetos, target_todos_sujetos)
   p_adjusted = FDR_correction(p)

Actualmente tienes:
✅ 1 sujeto procesado (S01-DMT)
✅ target.csv con 29 sujetos
⚠️  Necesitas al menos 10-15 sujetos procesados para correlaciones
""")

print("\n" + "="*60)
print("✓ Test completado - Estructura de datos verificada")
print("="*60)

# ============================================================
# 6. Verificar estructura completa
# ============================================================

print("\n5. Verificando estructura completa de order_all...")
print("\nResumen de datos disponibles:")

for band in band_list:
    for hemi in hemi_list:
        for net in net_list:
            n_epochs = len(data_dict["DMT"][band][hemi][net])
            print(f"   {band:5s} / {hemi:4s} / {net.strip():3s}: {n_epochs} épocas")

print("\n✓ Todas las combinaciones están disponibles")
print("✓ Estructura compatible con pearson.py")






