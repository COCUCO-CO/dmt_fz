#!/usr/bin/env python3
"""
Visualización de la Transformada de Hilbert como señal compleja en 3D
Muestra la señal analítica en el espacio (Real, Imaginario, Tiempo)
Similar a representaciones clásicas de señales complejas
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import sys

# Add parent directory to path to import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import BASE_DIR, RESULTS_DIR, RESULTS_PLOTS_DIR, ensure_dir

# ============================================================
# Funciones auxiliares
# ============================================================

def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

# ============================================================
# Configuración
# ============================================================

path = RESULTS_DIR
output_path = ensure_dir(RESULTS_PLOTS_DIR)

print("="*60)
print("VISUALIZACIÓN HILBERT - SEÑAL ANALÍTICA COMPLEJA")
print("="*60)

# ============================================================
# 1. Cargar datos
# ============================================================

print("\n1. Cargando datos...")

# Cargar metadata
[node_colors, label_names, label_names_short, 
 stc_coords_3d, ch_names, mapping, eeg_coords_2d] = load_file(path / "extra.pkl")

# Cargar datos del paciente
subject_file = path / "DMT" / "phases-S01-DMT.pkl"
subject_data = load_file(subject_file)

print(f"   ✓ Datos de S01-DMT cargados")

# ============================================================
# 2. Seleccionar región de ejemplo (DMN)
# ============================================================

band = "Alpha"
epoca = 0

# Encontrar una región del DMN (o la primera disponible)
dmn_roi = 0
dmn_name = label_names[0]

for i, name in enumerate(label_names[:100]):  # Buscar en las primeras 100
    if "Default" in name:  # Buscar DMN
        dmn_roi = i
        dmn_name = name
        break

print(f"\n2. Región seleccionada: {dmn_name} (ROI #{dmn_roi})")
print(f"   Banda: {band}, Época: {epoca}")

# Extraer señal de esa región
phases = subject_data["phases_stc"][band][epoca][dmn_roi, :]  # Serie temporal de fases
amplitudes = subject_data["amplitudes_stc"][band][epoca][dmn_roi, :]  # Serie temporal de amplitudes

# Reconstruir señal analítica: z(t) = A(t) * exp(i*phi(t))
analytical_signal = amplitudes * np.exp(1j * phases)
real_part = np.real(analytical_signal)
imag_part = np.imag(analytical_signal)

# Tiempo (asumiendo 500 Hz de sampling rate)
sampling_rate = 500
time = np.arange(len(phases)) / sampling_rate

print(f"   ✓ Señal analítica extraída")
print(f"   ✓ Duración: {time[-1]:.2f} segundos")
print(f"   ✓ Puntos temporales: {len(time)}")

# ============================================================
# 3. PLOT 1: Señal analítica en 3D (Real, Imag, Tiempo)
# ============================================================

print("\n3. Generando visualización 3D de señal analítica...")

fig = plt.figure(figsize=(16, 12))

# ============================================================
# Panel Superior: Vista 3D de la señal compleja
# ============================================================

ax1 = fig.add_subplot(221, projection='3d')

# Colorear por tiempo
colors = plt.cm.viridis(np.linspace(0, 1, len(time)))

# Plot de la trayectoria en el espacio complejo
for i in range(len(time)-1):
    ax1.plot(time[i:i+2], 
            real_part[i:i+2], 
            imag_part[i:i+2],
            color=colors[i],
            linewidth=2,
            alpha=0.7)

# Scatter para mostrar puntos
scatter = ax1.scatter(time, real_part, imag_part, 
                     c=time, cmap='viridis', 
                     s=20, alpha=0.6)

ax1.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Real(z)', fontsize=11, fontweight='bold')
ax1.set_zlabel('Imag(z)', fontsize=11, fontweight='bold')
ax1.set_title('Señal Analítica en Espacio Complejo\n(Tiempo, Real, Imaginario)', 
             fontsize=12, fontweight='bold')
ax1.view_init(elev=20, azim=45)

# ============================================================
# Vista lateral: Proyección Real vs Tiempo
# ============================================================

ax2 = fig.add_subplot(222)
ax2.plot(time, real_part, linewidth=1.5, color='steelblue', alpha=0.8)
ax2.fill_between(time, real_part, alpha=0.3, color='steelblue')
ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Real(z) = A·cos(φ)', fontsize=11, fontweight='bold')
ax2.set_title('Proyección: Parte Real', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# ============================================================
# Vista lateral: Proyección Imaginaria vs Tiempo
# ============================================================

ax3 = fig.add_subplot(223)
ax3.plot(time, imag_part, linewidth=1.5, color='coral', alpha=0.8)
ax3.fill_between(time, imag_part, alpha=0.3, color='coral')
ax3.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Imag(z) = A·sin(φ)', fontsize=11, fontweight='bold')
ax3.set_title('Proyección: Parte Imaginaria', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ============================================================
# Vista superior: Proyección en Plano Complejo
# ============================================================

ax4 = fig.add_subplot(224)
for i in range(len(real_part)-1):
    ax4.plot(real_part[i:i+2], imag_part[i:i+2], 
            color=colors[i], linewidth=2, alpha=0.7)

scatter4 = ax4.scatter(real_part, imag_part, 
                      c=time, cmap='viridis', 
                      s=30, alpha=0.6)
ax4.set_xlabel('Real(z)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Imag(z)', fontsize=11, fontweight='bold')
ax4.set_title('Proyección en Plano Complejo\n(trayectoria coloreada por tiempo)', 
             fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

# Colorbar
cbar = plt.colorbar(scatter, ax=[ax1, ax2, ax3, ax4], 
                   orientation='horizontal', 
                   pad=0.05, 
                   fraction=0.03,
                   label='Tiempo (s)')

fig.suptitle(f'Transformada de Hilbert - Señal Analítica Compleja\n' + 
            f'S01-DMT - {band} - {dmn_name.replace("7Networks_","").replace("-lh","").replace("-rh","")}',
            fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_path / f"hilbert_complex_signal_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'hilbert_complex_signal_S01-DMT_{band}.png'}")

# ============================================================
# 4. PLOT 2: Amplitud y Fase separadas
# ============================================================

print("\n4. Generando plot de amplitud y fase...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Amplitud (envolvente)
ax1.plot(time, amplitudes, linewidth=2, color='darkred', alpha=0.8)
ax1.fill_between(time, amplitudes, alpha=0.3, color='darkred')
ax1.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Amplitud (μV)', fontsize=11, fontweight='bold')
ax1.set_title('Envolvente (Amplitud Instantánea)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Fase instantánea
ax2.plot(time, phases, linewidth=2, color='purple', alpha=0.8)
ax2.fill_between(time, phases, alpha=0.3, color='purple')
ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Fase (radianes)', fontsize=11, fontweight='bold')
ax2.set_title('Fase Instantánea', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-np.pi, np.pi])
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

fig.suptitle(f'Componentes de la Transformada de Hilbert\n' + 
            f'S01-DMT - {band} - {dmn_name.replace("7Networks_","").replace("-lh","").replace("-rh","")}',
            fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_path / f"hilbert_amplitude_phase_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'hilbert_amplitude_phase_S01-DMT_{band}.png'}")

# ============================================================
# 5. PLOT 3: Múltiples regiones en plano complejo
# ============================================================

print("\n5. Generando comparación de múltiples regiones...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Trayectorias en Plano Complejo - Diferentes Regiones\n' + 
            f'S01-DMT - {band} - Época {epoca}',
            fontsize=14, fontweight='bold')

# Seleccionar 6 regiones de diferentes redes
selected_rois = []
networks = ["Default", "Cont", "DorsAttn", "Limbic", "SalVentAttn", "Vis"]
network_names = ["DMN", "FPN", "DAN", "LN", "SVA", "VN"]

for net in networks:
    for i, name in enumerate(label_names[:100]):
        if net in name:
            selected_rois.append((i, name, net))
            break

for idx, (roi_idx, roi_name, net) in enumerate(selected_rois):
    ax = axes[idx // 3, idx % 3]
    
    # Extraer señal
    phases_roi = subject_data["phases_stc"][band][epoca][roi_idx, :]
    amplitudes_roi = subject_data["amplitudes_stc"][band][epoca][roi_idx, :]
    
    # Reconstruir señal analítica
    analytical = amplitudes_roi * np.exp(1j * phases_roi)
    real = np.real(analytical)
    imag = np.imag(analytical)
    
    # Plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(real)))
    for i in range(len(real)-1):
        ax.plot(real[i:i+2], imag[i:i+2], 
               color=colors[i], linewidth=2, alpha=0.7)
    
    ax.scatter(real, imag, c=time, cmap='viridis', s=10, alpha=0.4)
    ax.set_xlabel('Real', fontsize=10)
    ax.set_ylabel('Imag', fontsize=10)
    ax.set_title(f'{network_names[idx]}: {roi_name[:30]}...', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.tight_layout()
plt.savefig(output_path / f"hilbert_multiple_regions_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'hilbert_multiple_regions_S01-DMT_{band}.png'}")

# ============================================================
# Resumen
# ============================================================

print("\n" + "="*60)
print("✓ VISUALIZACIÓN COMPLETADA")
print("="*60)
print(f"\nArchivos generados:")
print(f"  - hilbert_complex_signal_S01-DMT_{band}.png")
print(f"  - hilbert_amplitude_phase_S01-DMT_{band}.png")
print(f"  - hilbert_multiple_regions_S01-DMT_{band}.png")
print("\n" + "="*60)

