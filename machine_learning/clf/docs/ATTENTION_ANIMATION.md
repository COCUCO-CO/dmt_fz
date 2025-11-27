# Attention Animation Documentation

Script: `generate_attention_animation.py`

Genera videos que visualizan cómo evolucionan los patrones de atención de un modelo GAT (Graph Attention Network) a través de las épocas de EEG.

---

## Qué visualiza

El video muestra **dos grafos lado a lado** y **dos mini-mapas** que representan distintas métricas de atención:

### 1. Grafo Hub (izquierda, tonos rojos)

**Qué muestra:** Nodos que **emiten** atención hacia otros nodos.

**Cálculo:** Para cada nodo `i`, se suma la atención saliente:
```
hub_score[i] = Σⱼ attention[i → j]
```

**Interpretación:** Un nodo con alto puntaje Hub actúa como "broadcaster" — influye fuertemente sobre otros nodos del grafo. Las conexiones más gruesas/rojas indican mayor peso de atención saliente.

### 2. Grafo Selectivity (derecha, tonos verdes)

**Qué muestra:** Nodos que reciben atención de manera **selectiva** (no uniforme).

**Cálculo:** Basado en la entropía de la atención entrante:
```
Para cada nodo j:
  1. Normalizar atención entrante: α[i→j] = attention[i→j] / Σₖ attention[k→j]
  2. Calcular entropía: H[j] = -Σᵢ α[i→j] · log(α[i→j])
  3. Selectividad: selectivity[j] = 1 - H[j]/H_max
```

**Interpretación:** 
- **Alta selectividad (verde oscuro):** El nodo "escucha" principalmente a pocos vecinos específicos
- **Baja selectividad (verde claro):** El nodo recibe atención uniformemente de muchos vecinos

### 3. Mini-mapas (derecha)

Cada mini-mapa muestra una vista esquemática de la cabeza con los electrodos. 

- **Mini-mapa Hub (arriba):** Resalta en rojo los top-K nodos con mayor atención saliente
- **Mini-mapa Selectivity (abajo):** Resalta en verde los top-K nodos más selectivos
- **Rombo (◆):** Centro de masa ponderado de los top-K nodos. Indica la posición espacial promedio de la actividad
- **Estela:** Muestra la trayectoria reciente del centro de masa (últimos 50 frames)

### 4. Barras de color

- **Barra roja (Hub):** Escala de pesos de atención (valores absolutos)
- **Barra verde (Selectivity):** Porcentaje de selectividad (0% = uniforme, 100% = completamente selectivo)

### 5. Barra de progreso

Indica el progreso a través de las épocas de EEG.

---

## Parámetros importantes

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--checkpoint` | (requerido) | Ruta al modelo entrenado (.pt) |
| `--subject` | (requerido) | ID del sujeto (ej: `S01-DMT`) |
| `--layer` | 0 | Capa de atención GAT a visualizar (0-indexed) |
| `--top-k` | 3 | Número de nodos top para calcular centro de masa |
| `--fps` | 5 | Frames por segundo del video |
| `--transitions` | 1 | Frames de interpolación entre épocas (suaviza transiciones) |
| `--workers` | 1 | Procesos paralelos para renderizado |
| `--dpi` | 150 | Resolución (usar 100 para renderizado rápido) |
| `--edge-percentile` | 100 | Mostrar solo top N% de conexiones |
| `--mst` | false | Usar Minimum Spanning Tree en vez de grafo completo |
| `--conditions` | todas | Filtrar condiciones específicas |

---

## Uso

### Listar sujetos disponibles
```bash
python generate_attention_animation.py \
  --checkpoint path/to/best_model.pt \
  --subject dummy \
  --list-subjects
```

### Generar animación básica
```bash
python generate_attention_animation.py \
  --checkpoint path/to/best_model.pt \
  --subject S01-DMT
```

### Generar animación optimizada (rápida)
```bash
python generate_attention_animation.py \
  --checkpoint path/to/best_model.pt \
  --subject S01-DMT \
  --workers 8 \
  --dpi 100 \
  --fps 3
```

### Ver capa específica con centro de masa de 1 nodo
```bash
python generate_attention_animation.py \
  --checkpoint path/to/best_model.pt \
  --subject S01-DMT \
  --layer 2 \
  --top-k 1
```

### Generar solo para una condición
```bash
python generate_attention_animation.py \
  --checkpoint path/to/best_model.pt \
  --subject S01-DMT \
  --conditions DMT
```

---

## Output

El script genera archivos `.mp4` en el directorio especificado (default: `output/animations/`):

```
attention_animation_{subject}_{condition}_layer{N}.mp4
```

---

## Notas técnicas

1. **Atención en GATv2:** Los pesos de atención provienen de la capa `GATv2Conv`. Cada capa tiene múltiples cabezas de atención; el script promedia sobre todas las cabezas.

2. **Softmax por fila:** La matriz de atención tiene softmax aplicado por fila (nodo fuente), lo que significa que la suma de atención saliente de cada nodo es ~1. Por esto, el "Sink" tradicional (suma de atención entrante) no es informativo — se reemplazó por Selectivity basado en entropía.

3. **Interpolación:** Los frames de transición interpolan linealmente entre matrices de atención consecutivas para suavizar el video.

4. **Filtrado de edges:** Con grafos densos, se filtran automáticamente las conexiones débiles (percentil 30) para mejor visualización.

