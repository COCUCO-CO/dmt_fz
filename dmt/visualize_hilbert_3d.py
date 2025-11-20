#!/usr/bin/env python3
"""
Visualización 3D de la Transformada de Hilbert
Muestra fases y amplitudes en las parcelas del atlas Schaefer
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ============================================================
# Funciones auxiliares
# ============================================================

def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

# ============================================================
# Configuración
# ============================================================

path = Path("/home/tomas/PycharmProjects/dmt_fz/dmt/fwd-inv-stc/")
output_path = Path("/home/tomas/PycharmProjects/dmt_fz/dmt/results_plots/")
output_path.mkdir(exist_ok=True)

print("="*60)
print("VISUALIZACIÓN 3D - TRANSFORMADA DE HILBERT")
print("="*60)

# ============================================================
# 1. Cargar datos
# ============================================================

print("\n1. Cargando datos...")

# Cargar metadata (coordenadas 3D)
[node_colors, label_names, label_names_short, 
 stc_coords_3d, ch_names, mapping, eeg_coords_2d] = load_file(path / "extra.pkl")

print(f"   ✓ {len(label_names)} parcelas cargadas")
print(f"   ✓ Coordenadas 3D: {stc_coords_3d.shape}")

# Cargar datos del paciente
subject_file = path / "DMT" / "phases-S01-DMT.pkl"
subject_data = load_file(subject_file)

print(f"   ✓ Datos de S01-DMT cargados")

# ============================================================
# 2. Preparar datos para visualización
# ============================================================

band = "Alpha"
epoca = 0  # Primera época

print(f"\n2. Preparando visualización para banda {band}, época {epoca}...")

# Extraer fases y amplitudes
phases = subject_data["phases_stc"][band][epoca]  # (100 parcelas, tiempo)
amplitudes = subject_data["amplitudes_stc"][band][epoca]  # (100 parcelas, tiempo)

print(f"   ✓ Fases: {phases.shape}")
print(f"   ✓ Amplitudes: {amplitudes.shape}")

# Promediar en el tiempo para visualización estática
phase_mean = np.angle(np.exp(1j * phases).mean(axis=1))  # Fase circular promedio
amplitude_mean = amplitudes.mean(axis=1)  # Amplitud promedio

# Ajustar tamaños (las parcelas pueden tener dimensiones ligeramente diferentes)
n_coords = len(stc_coords_3d)
phase_mean = phase_mean[:n_coords]
amplitude_mean = amplitude_mean[:n_coords]

print(f"   ✓ Fase promedio: {phase_mean.shape}")
print(f"   ✓ Amplitud promedio: {amplitude_mean.shape}")
print(f"   ✓ Coordenadas: {stc_coords_3d.shape}")

# ============================================================
# 3. PLOT 1: Fases en 3D (colores representan fase)
# ============================================================

print("\n3. Generando Plot 1: Fases en 3D...")

fig = plt.figure(figsize=(16, 6))

# Vista 1: Lateral izquierda
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=phase_mean, 
                       cmap='hsv',  # HSV es ideal para fases circulares
                       s=100,
                       alpha=0.8,
                       vmin=-np.pi,
                       vmax=np.pi)
ax1.set_title("Vista Lateral Izquierda", fontsize=12, fontweight='bold')
ax1.set_xlabel("X (mm)")
ax1.set_ylabel("Y (mm)")
ax1.set_zlabel("Z (mm)")
ax1.view_init(elev=0, azim=180)

# Vista 2: Superior
ax2 = fig.add_subplot(132, projection='3d')
scatter2 = ax2.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=phase_mean, 
                       cmap='hsv',
                       s=100,
                       alpha=0.8,
                       vmin=-np.pi,
                       vmax=np.pi)
ax2.set_title("Vista Superior", fontsize=12, fontweight='bold')
ax2.set_xlabel("X (mm)")
ax2.set_ylabel("Y (mm)")
ax2.set_zlabel("Z (mm)")
ax2.view_init(elev=90, azim=0)

# Vista 3: Lateral derecha
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=phase_mean, 
                       cmap='hsv',
                       s=100,
                       alpha=0.8,
                       vmin=-np.pi,
                       vmax=np.pi)
ax3.set_title("Vista Lateral Derecha", fontsize=12, fontweight='bold')
ax3.set_xlabel("X (mm)")
ax3.set_ylabel("Y (mm)")
ax3.set_zlabel("Z (mm)")
ax3.view_init(elev=0, azim=0)

# Colorbar
cbar = plt.colorbar(scatter1, ax=[ax1, ax2, ax3], 
                    orientation='horizontal', 
                    pad=0.1, 
                    fraction=0.05,
                    label='Fase (radianes)')
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

fig.suptitle(f"Fases de Hilbert - S01-DMT - {band} - Época {epoca}", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path / f"hilbert_phases_3d_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'hilbert_phases_3d_S01-DMT_{band}.png'}")

# ============================================================
# 4. PLOT 2: Amplitudes en 3D (tamaño representa amplitud)
# ============================================================

print("\n4. Generando Plot 2: Amplitudes en 3D...")

fig = plt.figure(figsize=(16, 6))

# Normalizar amplitudes para visualización
amplitude_norm = (amplitude_mean - amplitude_mean.min()) / (amplitude_mean.max() - amplitude_mean.min())
sizes = 50 + amplitude_norm * 300  # Tamaños de 50 a 350

# Vista 1: Lateral izquierda
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=amplitude_mean, 
                       cmap='hot',
                       s=sizes,
                       alpha=0.7)
ax1.set_title("Vista Lateral Izquierda", fontsize=12, fontweight='bold')
ax1.set_xlabel("X (mm)")
ax1.set_ylabel("Y (mm)")
ax1.set_zlabel("Z (mm)")
ax1.view_init(elev=0, azim=180)

# Vista 2: Superior
ax2 = fig.add_subplot(132, projection='3d')
scatter2 = ax2.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=amplitude_mean, 
                       cmap='hot',
                       s=sizes,
                       alpha=0.7)
ax2.set_title("Vista Superior", fontsize=12, fontweight='bold')
ax2.set_xlabel("X (mm)")
ax2.set_ylabel("Y (mm)")
ax2.set_zlabel("Z (mm)")
ax2.view_init(elev=90, azim=0)

# Vista 3: Lateral derecha
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(stc_coords_3d[:, 0], 
                       stc_coords_3d[:, 1], 
                       stc_coords_3d[:, 2],
                       c=amplitude_mean, 
                       cmap='hot',
                       s=sizes,
                       alpha=0.7)
ax3.set_title("Vista Lateral Derecha", fontsize=12, fontweight='bold')
ax3.set_xlabel("X (mm)")
ax3.set_ylabel("Y (mm)")
ax3.set_zlabel("Z (mm)")
ax3.view_init(elev=0, azim=0)

# Colorbar
plt.colorbar(scatter1, ax=[ax1, ax2, ax3], 
            orientation='horizontal', 
            pad=0.1, 
            fraction=0.05,
            label='Amplitud (μV)')

fig.suptitle(f"Amplitudes de Hilbert - S01-DMT - {band} - Época {epoca}", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path / f"hilbert_amplitudes_3d_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'hilbert_amplitudes_3d_S01-DMT_{band}.png'}")

# ============================================================
# 5. PLOT 3: Sincronización entre regiones específicas
# ============================================================

print("\n5. Generando Plot 3: Sincronización de redes...")

# Identificar redes
df_labels = []
for name in label_names:
    hemi = name[:2]
    if "Cont" in name:
        net = "FPN"
    elif "Default" in name:
        net = "DMN"
    elif "DorsAttn" in name:
        net = "DAN"
    elif "Limbic" in name:
        net = "LN"
    elif "SalVentAttn" in name:
        net = "SVA"
    elif "SomMot" in name:
        net = "SMN"
    elif "Vis" in name:
        net = "VN"
    else:
        net = "Other"
    df_labels.append({"hemi": hemi, "net": net})

# Calcular sincronización (coherencia de fase) entre DMN y FPN
# Ajustar fases al número de coordenadas
phases_adjusted = phases[:n_coords, :]

dmn_idx = [i for i, label in enumerate(df_labels) if label["net"] == "DMN" and i < n_coords]
fpn_idx = [i for i, label in enumerate(df_labels) if label["net"] == "FPN" and i < n_coords]

print(f"   DMN: {len(dmn_idx)} regiones")
print(f"   FPN: {len(fpn_idx)} regiones")

# Fase promedio por red
dmn_phases = phases_adjusted[dmn_idx, :].mean(axis=0)
fpn_phases = phases_adjusted[fpn_idx, :].mean(axis=0)

# Diferencia de fase
phase_diff = np.abs(np.angle(np.exp(1j * (dmn_phases - fpn_phases))))

fig, ax = plt.subplots(figsize=(12, 4))
time_points = np.arange(len(phase_diff)) / 500  # Asumiendo 500 Hz
ax.plot(time_points, phase_diff, linewidth=2, color='purple')
ax.fill_between(time_points, phase_diff, alpha=0.3, color='purple')
ax.set_xlabel("Tiempo (s)", fontsize=12)
ax.set_ylabel("Diferencia de Fase (rad)", fontsize=12)
ax.set_title(f"Sincronización DMN-FPN - S01-DMT - {band} - Época {epoca}", 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, np.pi])

plt.tight_layout()
plt.savefig(output_path / f"network_sync_DMN_FPN_S01-DMT_{band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'network_sync_DMN_FPN_S01-DMT_{band}.png'}")

# ============================================================
# Resumen
# ============================================================

print("\n" + "="*60)
print("✓ VISUALIZACIÓN 3D COMPLETADA")
print("="*60)
print(f"\nResultados guardados en:")
print(f"  {output_path}")
print(f"\nArchivos generados:")
print(f"  - hilbert_phases_3d_S01-DMT_{band}.png")
print(f"  - hilbert_amplitudes_3d_S01-DMT_{band}.png")
print(f"  - network_sync_DMN_FPN_S01-DMT_{band}.png")
print("\n" + "="*60)

