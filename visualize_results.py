#!/usr/bin/env python3
"""
Script de visualización de resultados del pipeline
Genera plots de sincronización y Kuramoto
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")

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

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
conditions = ["DMT", "EC", "EO"]
colors = {"DMT": "red", "EC": "blue", "EO": "green"}

print("="*60)
print("VISUALIZACIÓN DE RESULTADOS")
print("="*60)

# ============================================================
# 1. Recopilar datos de Kuramoto
# ============================================================

print("\n1. Recopilando datos de Kuramoto (EEG y Sources)...")

kuramoto_data = {cond: {band: {"eeg": [], "stc": []} 
                        for band in band_list} 
                 for cond in conditions}

for cond in conditions:
    cond_path = path / cond
    files = sorted(cond_path.glob("phases-*.pkl"))
    
    print(f"\n   {cond}: {len(files)} archivos")
    
    for file in files:
        data = load_file(file)
        subject = file.stem.replace("phases-", "")
        
        for band in band_list:
            # Kuramoto EEG
            r_eeg = data["kuramoto_eeg"][band]
            kuramoto_data[cond][band]["eeg"].append({
                "subject": subject,
                "values": r_eeg,
                "mean": np.mean(r_eeg)
            })
            
            # Kuramoto Sources
            r_stc = data["kuramoto_stc"][band]
            kuramoto_data[cond][band]["stc"].append({
                "subject": subject,
                "values": r_stc,
                "mean": np.mean(r_stc)
            })

print("\n✓ Datos recopilados")

# ============================================================
# 2. PLOT 1: Kuramoto por banda y condición (EEG)
# ============================================================

print("\n2. Generando Plot 1: Kuramoto EEG por banda...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Parámetro de Orden de Kuramoto - EEG", fontsize=16, fontweight='bold')

for idx, band in enumerate(band_list):
    ax = axes[idx // 3, idx % 3]
    
    for cond in conditions:
        # Calcular media por sujeto
        means = [subj["mean"] for subj in kuramoto_data[cond][band]["eeg"]]
        subjects = [subj["subject"] for subj in kuramoto_data[cond][band]["eeg"]]
        
        ax.scatter(range(len(means)), means, label=cond, 
                  color=colors[cond], s=100, alpha=0.7)
        ax.plot(range(len(means)), means, color=colors[cond], 
               alpha=0.3, linewidth=2)
    
    ax.set_title(f"{band}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Sujeto")
    ax.set_ylabel("r (Kuramoto)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

# Ocultar subplot vacío
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_path / "kuramoto_eeg_by_band.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / 'kuramoto_eeg_by_band.png'}")

# ============================================================
# 3. PLOT 2: Kuramoto por banda y condición (Sources)
# ============================================================

print("\n3. Generando Plot 2: Kuramoto Sources por banda...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Parámetro de Orden de Kuramoto - Sources (Schaefer Atlas)", 
             fontsize=16, fontweight='bold')

for idx, band in enumerate(band_list):
    ax = axes[idx // 3, idx % 3]
    
    for cond in conditions:
        means = [subj["mean"] for subj in kuramoto_data[cond][band]["stc"]]
        subjects = [subj["subject"] for subj in kuramoto_data[cond][band]["stc"]]
        
        ax.scatter(range(len(means)), means, label=cond, 
                  color=colors[cond], s=100, alpha=0.7)
        ax.plot(range(len(means)), means, color=colors[cond], 
               alpha=0.3, linewidth=2)
    
    ax.set_title(f"{band}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Sujeto")
    ax.set_ylabel("r (Kuramoto)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_path / "kuramoto_sources_by_band.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / 'kuramoto_sources_by_band.png'}")

# ============================================================
# 4. PLOT 3: Evolución temporal de Kuramoto (ejemplo)
# ============================================================

print("\n4. Generando Plot 3: Evolución temporal...")

# Tomar un sujeto y banda de ejemplo
example_cond = "DMT"
example_band = "Alpha"
example_file = list((path / example_cond).glob("phases-*.pkl"))[0]
example_data = load_file(example_file)
example_subject = example_file.stem.replace("phases-", "")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle(f"Evolución Temporal de Kuramoto - {example_subject} - {example_band}", 
             fontsize=14, fontweight='bold')

# EEG
r_eeg = example_data["kuramoto_eeg"][example_band]
r_eeg_flat = np.array([np.mean(r) if hasattr(r, '__iter__') else r for r in r_eeg])
epochs = range(len(r_eeg_flat))
ax1.plot(epochs, r_eeg_flat, color='steelblue', linewidth=1, alpha=0.7)
ax1.axhline(np.mean(r_eeg_flat), color='red', linestyle='--', 
           label=f'Media: {np.mean(r_eeg_flat):.3f}')
ax1.fill_between(epochs, r_eeg_flat, alpha=0.3, color='steelblue')
ax1.set_title("EEG (24 canales)", fontsize=12)
ax1.set_xlabel("Época")
ax1.set_ylabel("r (Kuramoto)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Sources
r_stc = example_data["kuramoto_stc"][example_band]
r_stc_flat = np.array([np.mean(r) if hasattr(r, '__iter__') else r for r in r_stc])
epochs = range(len(r_stc_flat))
ax2.plot(epochs, r_stc_flat, color='darkorange', linewidth=1, alpha=0.7)
ax2.axhline(np.mean(r_stc_flat), color='red', linestyle='--', 
           label=f'Media: {np.mean(r_stc_flat):.3f}')
ax2.fill_between(epochs, r_stc_flat, alpha=0.3, color='darkorange')
ax2.set_title("Sources (100 parcelas Schaefer)", fontsize=12)
ax2.set_xlabel("Época")
ax2.set_ylabel("r (Kuramoto)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_path / f"kuramoto_temporal_{example_subject}_{example_band}.png", 
           dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / f'kuramoto_temporal_{example_subject}_{example_band}.png'}")

# ============================================================
# 5. PLOT 4: Comparación DMT vs Baseline (boxplot)
# ============================================================

print("\n5. Generando Plot 4: Comparación DMT vs Baseline...")

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Comparación Kuramoto: DMT vs Eyes-Closed vs Eyes-Open (Sources)", 
             fontsize=14, fontweight='bold')

for idx, band in enumerate(band_list):
    ax = axes[idx]
    
    data_for_box = []
    labels_for_box = []
    
    for cond in conditions:
        means = [subj["mean"] for subj in kuramoto_data[cond][band]["stc"]]
        data_for_box.append(means)
        labels_for_box.append(cond)
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Colorear boxes
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.6)
    
    ax.set_title(f"{band}", fontsize=12, fontweight='bold')
    ax.set_ylabel("r (Kuramoto)")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_path / "kuramoto_comparison_boxplot.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Guardado: {output_path / 'kuramoto_comparison_boxplot.png'}")

# ============================================================
# 6. Resumen estadístico
# ============================================================

print("\n6. Generando resumen estadístico...")

summary_data = []
for cond in conditions:
    for band in band_list:
        means_stc = [subj["mean"] for subj in kuramoto_data[cond][band]["stc"]]
        summary_data.append({
            "Condición": cond,
            "Banda": band,
            "Media": np.mean(means_stc),
            "SD": np.std(means_stc),
            "Min": np.min(means_stc),
            "Max": np.max(means_stc),
            "N": len(means_stc)
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_path / "summary_statistics.csv", index=False)
print(f"   ✓ Guardado: {output_path / 'summary_statistics.csv'}")

print("\n" + "="*60)
print("✓ VISUALIZACIÓN COMPLETADA")
print("="*60)
print(f"\nResultados guardados en:")
print(f"  {output_path}")
print(f"\nArchivos generados:")
print(f"  - kuramoto_eeg_by_band.png")
print(f"  - kuramoto_sources_by_band.png")
print(f"  - kuramoto_temporal_*.png")
print(f"  - kuramoto_comparison_boxplot.png")
print(f"  - summary_statistics.csv")
print("\n" + "="*60)

