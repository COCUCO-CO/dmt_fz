"""
Step metadata for pipeline UI.

Contains user-friendly names, descriptions, and phase groupings
for all pipeline steps. This is ONLY for display purposes -
it does NOT affect the actual execution logic.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List


class Phase(Enum):
    """Pipeline phases for visual grouping."""
    PROCESSING = "processing"
    CONNECTIVITY = "connectivity"
    ADVANCED = "advanced"


@dataclass
class PhaseInfo:
    """Information about a pipeline phase."""
    id: Phase
    title: str
    subtitle: str
    icon: str
    color: str
    collapsed_default: bool = False


@dataclass
class StepInfo:
    """User-friendly information about a pipeline step."""
    # Identity
    step_number: int
    script_name: str
    
    # Display
    display_name: str
    short_description: str
    detailed_description: str
    
    # Grouping
    phase: Phase
    
    # Timing
    time_estimate: str
    time_with_limit: str  # When using max_epochs or max_subjects
    
    # Output info
    output_description: str
    output_example: str
    
    # Dependencies
    requires_steps: List[int]  # Step numbers this depends on
    
    # Button
    button_text: str
    button_icon: str
    
    # File structure info (for detailed help)
    input_structure: str  # Detailed input file structure
    output_structure: str  # Detailed output file structure


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

PHASES = {
    Phase.PROCESSING: PhaseInfo(
        id=Phase.PROCESSING,
        title="PROCESAMIENTO INICIAL",
        subtitle="Transformación de señales EEG a actividad cerebral",
        icon="memory",
        color="#00d4aa",  # Primary green
    ),
    Phase.CONNECTIVITY: PhaseInfo(
        id=Phase.CONNECTIVITY,
        title="ANÁLISIS DE CONECTIVIDAD",
        subtitle="Sincronización y coherencia entre regiones",
        icon="hub",
        color="#f59e0b",  # Orange/amber
    ),
    Phase.ADVANCED: PhaseInfo(
        id=Phase.ADVANCED,
        title="ANÁLISIS AVANZADO",
        subtitle="Correlaciones y estados cerebrales (opcional)",
        icon="analytics",
        color="#a78bfa",  # Purple
        collapsed_default=True,
    ),
}


# =============================================================================
# STEP DEFINITIONS
# =============================================================================

STEP_INFO = {
    1: StepInfo(
        step_number=1,
        script_name="fwd.py",
        display_name="Localización de Fuentes",
        short_description="EEG de sensores → actividad en 100 regiones cerebrales",
        detailed_description="""
Transforma las señales EEG registradas en los electrodos del cuero cabelludo 
en estimaciones de actividad neuronal en 100 regiones cerebrales (parcelas del 
atlas Schaefer). Utiliza un modelo de cabeza estándar y la solución inversa 
eLORETA para esta transformación.

Para cada época de EEG, extrae las fases instantáneas usando la transformada 
de Hilbert en 5 bandas de frecuencia (Delta, Theta, Alpha, Beta, Gamma).
        """.strip(),
        phase=Phase.PROCESSING,
        time_estimate="~3-4 horas (35 sujetos, todas las épocas)",
        time_with_limit="~30-45 min con max_epochs=50",
        output_description="Un archivo por sujeto con las fases cerebrales",
        output_example="phases-S01-DMT.pkl, phases-S02-DMT.pkl...",
        requires_steps=[],
        button_text="Ejecutar Localización",
        button_icon="psychology",
        input_structure="""
📂 ENTRADA: Archivos .set (EEGLAB)
─────────────────────────────────
EEG_CLEAN/
├── DMT/
│   ├── S01-DMT_ICA_pruned.set  ← Archivo principal
│   ├── S01-DMT_ICA_pruned.fdt  ← Datos binarios
│   └── ...
├── EC/
└── EO/

Cada .set contiene:
• Épocas segmentadas (duración variable)
• n_channels canales EEG
• Frecuencia de muestreo del dataset
• Datos ya preprocesados con ICA
        """.strip(),
        output_structure="""
📂 SALIDA: phases-{sujeto}-{cond}.pkl
─────────────────────────────────────
{
  'phases_stc': {           # Fases en espacio fuente
    'Delta': ndarray[n_epochs, 100, n_times],
    'Theta': ndarray[n_epochs, 100, n_times],
    'Alpha': ndarray[n_epochs, 100, n_times],
    'Beta':  ndarray[n_epochs, 100, n_times],
    'Gamma': ndarray[n_epochs, 100, n_times]
  },
  'phases_eeg': {           # Fases en espacio sensor
    'Delta': ndarray[n_epochs, n_channels, n_times],
    ...
  },
  'sfreq': <freq_original>, # Frecuencia del dataset
  'subject': 'S01-DMT',     # ID del sujeto
  'n_epochs': <n_epochs>    # Cantidad de épocas
}

Dimensiones:
• n_epochs: cantidad de épocas del sujeto
• 100 parcelas (Schaefer atlas) - fijo
• n_channels: canales del dataset original
• n_times: muestras por época
        """.strip(),
    ),
    
    2: StepInfo(
        step_number=2,
        script_name="save_load_pickle.py",
        display_name="Consolidar Datos",
        short_description="Agrupa archivos individuales → un archivo por condición",
        detailed_description="""
Combina todos los archivos phases-*.pkl individuales de cada sujeto en 
archivos consolidados por condición experimental (DMT, EC, EO).

Esto facilita el análisis posterior al tener todos los datos de una 
condición en un solo archivo.
        """.strip(),
        phase=Phase.PROCESSING,
        time_estimate="~1-2 minutos",
        time_with_limit="~1-2 minutos",
        output_description="Un archivo consolidado por condición",
        output_example="subject_phases_DMT.pkl, subject_phases_EC.pkl...",
        requires_steps=[1],
        button_text="Consolidar Archivos",
        button_icon="folder_copy",
        input_structure="""
📂 ENTRADA: phases-*.pkl (del Step 1)
─────────────────────────────────────
run_*/
├── DMT/
│   ├── phases-S01-DMT.pkl
│   ├── phases-S02-DMT.pkl
│   └── ...
├── EC/
│   └── phases-S01-EC.pkl ...
└── EO/
    └── phases-S01-EO.pkl ...

Cada archivo contiene las fases de UN sujeto.
        """.strip(),
        output_structure="""
📂 SALIDA: subject_phases_{cond}.pkl
────────────────────────────────────
{
  'S01-DMT': {              # Por sujeto
    'phases_stc': {...},
    'phases_eeg': {...},
    'sfreq': <freq>,
    'n_epochs': <n>
  },
  'S02-DMT': {...},
  ...
}

Un diccionario con TODOS los sujetos de esa condición.
Facilita iterar sobre sujetos para análisis grupales.
        """.strip(),
    ),
    
    3: StepInfo(
        step_number=3,
        script_name="multi2pool2.py",
        display_name="Filtrar por Redes",
        short_description="Separa actividad según las 7 redes cerebrales funcionales",
        detailed_description="""
Filtra los datos de las 100 regiones cerebrales según las 7 redes funcionales 
definidas en el atlas Schaefer:

• Visual (Vis) - Procesamiento visual
• Somatomotor (SomMot) - Control motor y sensorial
• Dorsal Attention (DorsAttn) - Atención dirigida
• Ventral Attention (SalVentAttn) - Detección de estímulos salientes
• Limbic - Procesamiento emocional
• Control (Cont) - Control ejecutivo
• Default Mode (Default) - Pensamiento interno, memoria

Permite analizar la conectividad dentro y entre redes específicas.
        """.strip(),
        phase=Phase.CONNECTIVITY,
        time_estimate="~10-20 minutos",
        time_with_limit="~5-10 minutos",
        output_description="Datos filtrados por red cerebral",
        output_example="network_filtered_DMT.pkl...",
        requires_steps=[2],
        button_text="Filtrar Redes",
        button_icon="hub",
        input_structure="""
📂 ENTRADA: subject_phases_{cond}.pkl
─────────────────────────────────────
Archivo consolidado con phases_stc de 100 parcelas.

Las 100 parcelas corresponden a:
• 50 hemisferio izquierdo
• 50 hemisferio derecho
Distribuidas en 7 redes funcionales (Schaefer atlas).
        """.strip(),
        output_structure="""
📂 SALIDA: Datos filtrados por red
──────────────────────────────────
{
  'Vis': {                  # Red Visual
    'indices': [...],       # Índices de parcelas
    'data': ndarray
  },
  'SomMot': {...},          # Somatomotor
  'DorsAttn': {...},        # Dorsal Attention
  'SalVentAttn': {...},     # Ventral Attention  
  'Limbic': {...},          # Limbic
  'Cont': {...},            # Control
  'Default': {...}          # Default Mode Network
}

Las 7 redes del atlas Schaefer-100:
• Vis, SomMot, DorsAttn, SalVentAttn
• Limbic, Cont, Default
(cantidad de parcelas según atlas)
        """.strip(),
    ),
    
    4: StepInfo(
        step_number=4,
        script_name="calculate_syncro.py",
        display_name="Sincronización",
        short_description="Calcula conectividad funcional entre regiones",
        detailed_description="""
Calcula matrices de sincronización de fase entre todas las regiones 
cerebrales. Utiliza el Phase Locking Value (PLV) para medir cuánto 
se sincronizan dos regiones a lo largo del tiempo.

Genera matrices de conectividad para cada banda de frecuencia y 
cada época, permitiendo analizar patrones de comunicación cerebral.
        """.strip(),
        phase=Phase.CONNECTIVITY,
        time_estimate="~20-40 minutos",
        time_with_limit="~10-15 minutos",
        output_description="Matrices de sincronización por época y banda",
        output_example="syncro-S01-DMT.pkl...",
        requires_steps=[3],
        button_text="Calcular Sincronización",
        button_icon="sync",
        input_structure="""
📂 ENTRADA: Fases del Step anterior
───────────────────────────────────
Datos de phases_stc con fases instantáneas:
• Valores en rango [-π, π]
• Una matriz por banda de frecuencia
• Dimensiones: [n_epochs, n_parcelas, n_times]
        """.strip(),
        output_structure="""
📂 SALIDA: syncro-{sujeto}-{cond}.pkl
─────────────────────────────────────
{
  'syncros_stc': {          # Sincronización fuente
    'Delta': {
      'epoch0': ndarray[100, 100],  # Matriz PLV
      'epoch1': ndarray[100, 100],
      ...
    },
    'Theta': {...},
    'Alpha': {...},
    'Beta': {...},
    'Gamma': {...}
  },
  'syncros_eeg': {          # Sincronización sensor
    'Delta': {...},
    ...
  }
}

Cada matriz [100, 100] o [n_ch, n_ch]:
• PLV entre pares de regiones/canales
• Valores en rango [0, 1]
• 0 = sin sincronización
• 1 = perfectamente sincronizados
• Matriz simétrica
        """.strip(),
    ),
    
    5: StepInfo(
        step_number=5,
        script_name="generate_order.py",
        display_name="Coherencia Global",
        short_description="Mide el orden/coherencia global usando Kuramoto",
        detailed_description="""
Calcula el parámetro de orden de Kuramoto, que mide cuán sincronizado 
está el cerebro globalmente. 

• Valores cercanos a 0: Actividad desorganizada/asincrónica
• Valores cercanos a 1: Alta sincronización global

Este parámetro es especialmente relevante para estudiar estados 
alterados de consciencia, donde típicamente se observan cambios 
en la coherencia global.
        """.strip(),
        phase=Phase.CONNECTIVITY,
        time_estimate="~5-10 minutos",
        time_with_limit="~2-5 minutos",
        output_description="Parámetro de orden por época",
        output_example="order-S01-DMT.pkl...",
        requires_steps=[4],
        button_text="Calcular Kuramoto",
        button_icon="waves",
        input_structure="""
📂 ENTRADA: Fases instantáneas
──────────────────────────────
phases_stc con fases en cada timepoint:
• θᵢ(t) = fase de la región i en tiempo t
• Valores en [-π, π]
        """.strip(),
        output_structure="""
📂 SALIDA: order-{sujeto}-{cond}.pkl
────────────────────────────────────
{
  'order_stc': {            # Orden en espacio fuente
    'Delta': {
      'epoch0': ndarray[n_times],  # R(t) por tiempo
      'epoch1': ndarray[n_times],
      ...
    },
    'Theta': {...},
    ...
  },
  'order_eeg': {...}        # Orden en espacio sensor
}

Parámetro de Kuramoto R(t):
  R(t) = |1/N Σ exp(i·θⱼ(t))|

• R ≈ 0: Fases aleatorias (desorden)
• R ≈ 1: Fases alineadas (orden total)
        """.strip(),
    ),
    
    6: StepInfo(
        step_number=6,
        script_name="build_order_data.py",
        display_name="Agregar Métricas",
        short_description="Combina métricas de todos los sujetos",
        detailed_description="""
Agrega las métricas de coherencia (Kuramoto) de todos los sujetos 
en estructuras de datos unificadas para facilitar el análisis 
estadístico y las comparaciones entre condiciones.

Genera resúmenes por condición, sujeto y banda de frecuencia.
        """.strip(),
        phase=Phase.CONNECTIVITY,
        time_estimate="~2-5 minutos",
        time_with_limit="~1-2 minutos",
        output_description="Datos agregados de todos los sujetos",
        output_example="order_all_DMT.pkl, kuramoto_summary.pkl...",
        requires_steps=[5],
        button_text="Agregar Datos",
        button_icon="merge",
        input_structure="""
📂 ENTRADA: order-*.pkl (del Step 5)
────────────────────────────────────
run_*/
├── DMT/
│   ├── order-S01-DMT.pkl
│   ├── order-S02-DMT.pkl
│   └── ...
├── EC/
└── EO/
        """.strip(),
        output_structure="""
📂 SALIDA: order_all_{cond}.pkl
───────────────────────────────
{
  'subjects': ['S01', 'S02', ...],
  'bands': ['Delta', 'Theta', ...],
  'order_mean': {           # Media por sujeto
    'S01': {
      'Delta': <valor>,     # R promedio
      'Theta': <valor>,
      ...
    },
    ...
  },
  'order_std': {...},       # Desviación estándar
  'order_timeseries': {...} # Series temporales
}

Archivos adicionales (si se generan):
• Resúmenes CSV
• Visualizaciones PNG
        """.strip(),
    ),
    
    7: StepInfo(
        step_number=7,
        script_name="pearson.py",
        display_name="Correlaciones",
        short_description="Analiza correlaciones entre condiciones",
        detailed_description="""
Calcula correlaciones de Pearson entre las matrices de conectividad 
de diferentes condiciones experimentales.

Permite identificar:
• Similitudes/diferencias en patrones de conectividad
• Regiones que cambian significativamente entre condiciones
• Correlaciones con medidas conductuales o subjetivas

⚠️ Requiere múltiples sujetos para resultados estadísticamente válidos.
        """.strip(),
        phase=Phase.ADVANCED,
        time_estimate="~30 min - 2 horas (depende de sujetos)",
        time_with_limit="~15-30 minutos",
        output_description="Matrices de correlación entre condiciones",
        output_example="pearson_results/*.pkl",
        requires_steps=[4],
        button_text="Calcular Correlaciones",
        button_icon="compare_arrows",
        input_structure="""
📂 ENTRADA: syncro-*.pkl (del Step 4)
─────────────────────────────────────
Matrices de sincronización PLV:
• Una matriz por época
• Valores entre 0 y 1
• Múltiples sujetos y condiciones
        """.strip(),
        output_structure="""
📂 SALIDA: pearson_results/
───────────────────────────
pearson_results/
├── correlation_{cond1}_vs_{cond2}.pkl
└── ...

Cada archivo contiene:
{
  'r_values': ndarray[n, n],    # Correlación r
  'p_values': ndarray[n, n],    # Significancia p
  'significant': ndarray[n, n], # Bool mask (p<0.05)
  'mean_r': <valor>,            # Correlación media
  'conditions': (cond1, cond2)
}

Interpretación:
• r > 0.7: Alta correlación (similar)
• r < 0.3: Baja correlación (diferente)
• p < 0.05: Estadísticamente significativo
        """.strip(),
    ),
    
    8: StepInfo(
        step_number=8,
        script_name="clustering.py",
        display_name="Estados Cerebrales",
        short_description="Identifica patrones/estados recurrentes mediante clustering",
        detailed_description="""
Aplica algoritmos de clustering (K-means) a las matrices de conectividad 
para identificar "estados cerebrales" recurrentes.

Cada estado representa un patrón de conectividad que aparece 
repetidamente a lo largo del tiempo. Permite analizar:

• Cuántos estados distintos existen
• Cuánto tiempo se permanece en cada estado
• Transiciones entre estados
• Diferencias en estados entre condiciones

Útil para caracterizar la dinámica cerebral bajo diferentes condiciones.
        """.strip(),
        phase=Phase.ADVANCED,
        time_estimate="Quick: ~30 min, Full: ~4 horas",
        time_with_limit="Quick: ~15-30 min",
        output_description="Clusters de estados cerebrales",
        output_example="clustering_results/*.pkl, *.png",
        requires_steps=[4],
        button_text="Identificar Estados",
        button_icon="category",
        input_structure="""
📂 ENTRADA: syncro-*.pkl (del Step 4)
─────────────────────────────────────
Matrices PLV aplanadas para clustering:
• Triángulo superior de matriz de conectividad
• Un vector por época
• Todos los sujetos concatenados
        """.strip(),
        output_structure="""
📂 SALIDA: clustering_results/
──────────────────────────────
clustering_results/
├── kmeans_k{k}_pca{n}.pkl      # Modelo entrenado
├── cluster_labels.pkl          # Etiquetas por época
├── cluster_centers.pkl         # Centroides
├── silhouette_scores.csv       # Métricas calidad
└── visualizations/
    └── *.png                   # Gráficos generados

Estructura cluster_labels.pkl:
{
  'labels': ndarray[n_epochs],  # 0,1,2...k-1
  'subjects': [...],
  'conditions': [...],
  'epochs': [...]
}

Métricas por cluster:
• Occupancy: % tiempo en cada estado
• Dwell time: duración promedio
• Transitions: matriz de transiciones
        """.strip(),
    ),
}


def get_steps_for_phase(phase: Phase) -> List[StepInfo]:
    """Get all steps belonging to a phase."""
    return [info for info in STEP_INFO.values() if info.phase == phase]


def get_step_by_script(script_name: str) -> StepInfo:
    """Get step info by script name."""
    for info in STEP_INFO.values():
        if info.script_name == script_name:
            return info
    return None

