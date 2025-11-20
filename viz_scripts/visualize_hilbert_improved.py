#!/usr/bin/env python3
"""
Visualización de la señal analítica (Hilbert) con plotly.
Genera UNA figura 3D limpia con las proyecciones ortogonales y exporta
tanto en PNG como en HTML interactivo.
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

# Add parent directory to path to import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import BASE_DIR, RESULTS_DIR, RESULTS_PLOTS_DIR, ensure_dir


def load_file(file):
    with open(file, "rb") as handle:
        return pickle.load(handle)


# =============================================================================
# Configuración
# =============================================================================

path = RESULTS_DIR
output_path = ensure_dir(RESULTS_PLOTS_DIR)

subject = "S01-DMT"
condition = "DMT"
band = "Gamma"
epoch_idx = 10

print("=" * 70)
print("Visualización señal analítica 3D (Plotly)")
print("=" * 70)


# =============================================================================
# 1. Cargar datos
# =============================================================================

print("1) Cargando datos...")
[node_colors, label_names, label_names_short,
 stc_coords_3d, ch_names, mapping, eeg_coords_2d] = load_file(path / "extra.pkl")

subject_file = path / condition / f"phases-{subject}.pkl"
subject_data = load_file(subject_file)

print(f"   ✓ Dataset: {subject_file.name}")


# =============================================================================
# 2. Seleccionar ROI y época
# =============================================================================

roi_idx = 0
roi_name = label_names[0]
for idx, name in enumerate(label_names[:120]):
    if "Default" in name or "Cont" in name:
        roi_idx = idx
        roi_name = name
        break

epochs = subject_data["phases_stc"][band]
num_epochs = len(epochs)

print(f"2) Selección:")
print(f"   • ROI: {roi_name} (idx={roi_idx})")
print(f"   • Época: {epoch_idx + 1}/{num_epochs}")

phases = subject_data["phases_stc"][band][epoch_idx][roi_idx, :]
amplitudes = subject_data["amplitudes_stc"][band][epoch_idx][roi_idx, :]

fs = 500.0
time = np.arange(len(phases)) / fs


# =============================================================================
# 3. Recorte central (evitar bordes)
# =============================================================================

start = int(len(time) * 0.15)
end = int(len(time) * 0.85)

time_seg = time[start:end]
phase_seg = phases[start:end]
amp_seg = amplitudes[start:end]

analytic_seg = amp_seg * np.exp(1j * phase_seg)
real_seg = analytic_seg.real
imag_seg = analytic_seg.imag

print("3) Segmento utilizado:")
print(f"   • Duración: {time_seg[-1] - time_seg[0]:.3f} s")
print(f"   • Puntos: {len(time_seg)}")


# =============================================================================
# 4. Construcción figura Plotly
# =============================================================================

print("4) Construyendo figura 3D...")

# Pista principal (tiempo vs real vs imag)
trajectory = go.Scatter3d(
    x=time_seg,
    y=real_seg,
    z=imag_seg,
    mode="lines",
    line=dict(
        color=time_seg,
        colorscale="Viridis",
        width=4,
        colorbar=dict(title="Tiempo (s)")
    ),
    name="Trayectoria"
)


def projection_trace(x, y, z, name, color, dash="dot"):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=2.0, dash=dash),
        hoverinfo="skip",
        showlegend=True,
        name=name,
        opacity=0.65
    )


z_floor = np.full_like(time_seg, imag_seg.min() - 0.02)
proj_time_real = projection_trace(
    time_seg,
    real_seg,
    z_floor,
    "Proyección tiempo-real",
    "rgba(31, 119, 180, 0.75)",
    dash="dot"
)

y_wall = np.full_like(time_seg, real_seg.min() - 0.02)
proj_time_imag = projection_trace(
    time_seg,
    y_wall,
    imag_seg,
    "Proyección tiempo-imag",
    "rgba(214, 39, 40, 0.8)",
    dash="dot"
)

x_wall = np.full_like(real_seg, time_seg.max() + 0.02)
proj_real_imag = projection_trace(
    x_wall,
    real_seg,
    imag_seg,
    "Proyección plano complejo",
    "rgba(148, 103, 189, 0.8)",
    dash="longdash"
)


# Marcadores inicio/fin
markers = go.Scatter3d(
    x=[time_seg[0], time_seg[-1]],
    y=[real_seg[0], real_seg[-1]],
    z=[imag_seg[0], imag_seg[-1]],
    mode="markers",
    marker=dict(size=6, color=["#2ca02c", "#ff7f0e"], symbol=["circle", "square"],
                line=dict(width=1.5, color="black")),
    name="Inicio / Final",
    hovertemplate="Tiempo: %{x:.3f}s<br>Real: %{y:.4f}<br>Imag: %{z:.4f}<extra></extra>"
)


fig = go.Figure(data=[trajectory, proj_time_real, proj_time_imag,
                      proj_real_imag, markers])

fig.update_layout(
    width=1400,
    height=900,
    template="plotly_white",
    scene=dict(
        xaxis=dict(
            title=dict(text="Tiempo (s)", font=dict(size=18)),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title=dict(text="Real(z) = A·cos(φ)", font=dict(size=18)),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(0,0,0,0.1)"
        ),
        zaxis=dict(
            title=dict(text="Imag(z) = A·sin(φ)", font=dict(size=18)),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(0,0,0,0.1)"
        ),
        camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        aspectmode="manual",
        aspectratio=dict(x=1.5, y=1, z=1)
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        font=dict(size=13)
    ),
    margin=dict(l=20, r=20, b=20, t=100),
    title=dict(
        text=(
            "Transformada de Hilbert - Señal Analítica 3D"
            f"<br><span style='font-size:14px;'>"
            f"{subject} | Condición {condition} | Banda {band} | Época #{epoch_idx + 1}"
            f" | ROI: {roi_name}</span>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=24)
    )
)

print("   ✓ Figura construida")


# =============================================================================
# 5. Guardar archivos
# =============================================================================

png_path = output_path / f"hilbert_plotly_{subject}_{band}_epoch{epoch_idx+1}.png"
html_path = output_path / f"hilbert_plotly_{subject}_{band}_epoch{epoch_idx+1}.html"

fig.write_html(html_path)
print(f"   ✓ Figura interactiva: {html_path}")

try:
    import plotly.io as pio

    pio.write_image(fig, png_path, width=1400, height=900, scale=2)
    print(f"   ✓ PNG de alta resolución: {png_path}")
except Exception as exc:  # pragma: no cover
    print("   ⚠️ No se pudo exportar PNG. Instala 'kaleido'.")
    print(f"     Detalle: {exc}")


print("=" * 70)
print("Proceso finalizado")
print("=" * 70)

