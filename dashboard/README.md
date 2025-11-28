# 🧠 DMT-EEG Analysis Dashboard

Dashboard integrado para el análisis de sincronización EEG y clasificación de estados cerebrales.

## 🚀 Quick Start

```bash
# Activar ambiente conda
conda activate dmt_fz

# Iniciar dashboard
cd /media/storage_hdd/dmt_fz/dashboard
./run_dashboard.sh

# O directamente con streamlit
streamlit run app.py
```

El dashboard estará disponible en: **http://localhost:8501**

## 📋 Páginas

| Página | Descripción |
|--------|-------------|
| 🏠 **Home** | Vista general, estado del pipeline, métricas |
| 📊 **Data Explorer** | Explorar datos EEG por sujeto/condición/banda |
| 🔬 **Pipeline Runner** | Ejecutar scripts del pipeline con UI |
| 🤖 **ML Training** | Entrenar modelos GAT, hyperparameter search |
| 📈 **Experiments** | Ver y comparar experimentos de ML |
| 🎨 **Visualizations** | Attention heatmaps, MST, análisis |
| 🎬 **Animations** | Generar videos de atención temporal |

## ⚙️ Configuración

Edita `config.yaml` para ajustar rutas y parámetros:

```yaml
paths:
  project_root: "/media/storage_hdd/dmt_fz"
  eeg_clean: "/media/storage_hdd/dmt_fz/EEG_CLEAN"
  fwd_inv_stc: "/media/storage_hdd/dmt_fz/fwd-inv-stc"
  # ...

data:
  conditions: ["DMT", "EC", "EO"]
  bands: ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
```

## 📦 Dependencias

```bash
pip install streamlit plotly pandas pyyaml
```

Otras dependencias del proyecto (`torch`, `mne`, etc.) ya deben estar instaladas en el ambiente `dmt_fz`.

## 🗂️ Estructura

```
dashboard/
├── app.py                  # Entry point
├── config.yaml             # Configuración
├── run_dashboard.sh        # Script de inicio
├── pages/                  # Páginas de Streamlit
│   ├── 01_🏠_Home.py
│   ├── 02_📊_Data_Explorer.py
│   ├── 03_🔬_Pipeline_Runner.py
│   ├── 04_🤖_ML_Training.py
│   ├── 05_📈_Experiments.py
│   ├── 06_🎨_Visualizations.py
│   └── 07_🎬_Animations.py
├── backend/                # Lógica de negocio
├── components/             # Componentes reutilizables
├── database/               # Base de datos local
└── static/                 # Archivos estáticos
```

## 💡 Tips

- **TensorBoard**: Para monitoreo detallado del entrenamiento, abre TensorBoard en paralelo:
  ```bash
  tensorboard --logdir=machine_learning/clf/runs --port=6006
  ```

- **Memoria**: Si hay problemas de memoria al explorar datos grandes, reduce el número de épocas visualizadas

- **Puerto ocupado**: Si el puerto 8501 está en uso, el script automáticamente intenta el 8502

## 🐛 Troubleshooting

### Error: "No module named streamlit"
```bash
pip install streamlit plotly
```

### Gráficos no se muestran
Asegúrate de tener `plotly` instalado:
```bash
pip install plotly
```

### Datos no encontrados
Verifica las rutas en `config.yaml` y asegúrate de que apuntan a los directorios correctos.

---

**Última actualización:** 2025-11-25









