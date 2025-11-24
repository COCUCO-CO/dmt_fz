# Guía de Configuración de Grafos

## 🎯 Configuración Simple

Tu archivo `config/config.yaml` tiene **2 opciones principales**:

---

## 1️⃣ **¿Qué Usar Como Nodos?**

```yaml
data:
  use_stc: true   # o false
```

### **`use_stc: true`** ← RECOMENDADO
- **Nodos:** 100 parcelas Schaefer (solución inversa dSPM)
- **Ventaja:** Interpretabilidad neuroanatómica clara
- **Usa:** `syncros_stc`, `phases_stc`, `kuramoto_stc` de tus `phases-*.pkl`

### **`use_stc: false`**
- **Nodos:** 24 canales EEG (electrodos)
- **Ventaja:** Más directo, sin asumir modelo inverso perfecto
- **Usa:** `syncros_eeg`, `phases_eeg`, `kuramoto_eeg` de tus `phases-*.pkl`

**Cómo cambiar:**
```yaml
# Para usar electrodos:
use_stc: false

# Para usar source space (parcelas):
use_stc: true
```

---

## 2️⃣ **¿Fully Connected o Threshold?**

```yaml
data:
  graph:
    fully_connected: true   # o false
```

### **`fully_connected: true`** ← AHORA POR DEFECTO
- **Aristas:** TODAS las conexiones posibles
- **Cantidad:** N × (N-1) aristas
  - Con 100 nodos: ~10,000 aristas
  - Con 24 nodos: ~552 aristas
- **Filosofía:** La GAT decide qué conexiones importan vía attention
- **Pros:** Sin sesgos, máxima flexibilidad, la red aprende todo
- **Contras:** Puede aprender ruido en conexiones débiles si dataset pequeño

### **`fully_connected: false`**
- **Aristas:** Solo donde `syncro_matrix[i,j] > threshold`
- **Threshold:** Configurable con `edge_threshold: 0.3`
- **Filosofía:** Impones conocimiento del dominio (sync débil = ruido)
- **Pros:** Más robusto con datasets pequeños, grafos interpretables
- **Contras:** Pierdes información, hiperparámetro adicional

**Cómo cambiar:**
```yaml
# Fully connected (red aprende todo):
graph:
  fully_connected: true

# Con threshold (solo conexiones fuertes):
graph:
  fully_connected: false
  edge_threshold: 0.3  # 30% de sincronización mínima
```

---

## 🔧 **Configuraciones Típicas**

### **Setup 1: Source Space + Fully Connected** ← ACTUAL
```yaml
data:
  use_stc: true
  graph:
    fully_connected: true
```
- **Resultado:** 100 nodos (parcelas), ~10,000 aristas
- **Uso:** Máxima capacidad, dejas que la red aprenda
- **RTX 4090:** Sin problemas

### **Setup 2: Source Space + Threshold**
```yaml
data:
  use_stc: true
  graph:
    fully_connected: false
    edge_threshold: 0.3
```
- **Resultado:** 100 nodos, ~2,000-4,000 aristas (depende de sync)
- **Uso:** Más conservador, mejor para datasets pequeños

### **Setup 3: Electrodos + Fully Connected**
```yaml
data:
  use_stc: false
  graph:
    fully_connected: true
```
- **Resultado:** 24 nodos, ~552 aristas
- **Uso:** Rápido, sin asumir modelo inverso, totalmente conectado

### **Setup 4: Electrodos + Threshold**
```yaml
data:
  use_stc: false
  graph:
    fully_connected: false
    edge_threshold: 0.3
```
- **Resultado:** 24 nodos, ~100-200 aristas
- **Uso:** Más sparse, para comparar con EEGNet

---

## 📊 **Comparación Visual**

```
use_stc: true (Source Space)
├── 100 nodos (parcelas Schaefer)
├── FPN, DMN, DAN, etc. identificables
└── fully_connected: true → ~10,000 aristas
    fully_connected: false → ~2,000-4,000 aristas (con threshold 0.3)

use_stc: false (Electrodos)
├── 24 nodos (canales EEG)
├── Posiciones de electrodos
└── fully_connected: true → ~552 aristas
    fully_connected: false → ~100-200 aristas (con threshold 0.3)
```

---

## 🎓 **Recomendaciones**

### **Para Exploración Inicial:**
```yaml
use_stc: true          # Mejor interpretabilidad
fully_connected: true  # Deja que la red decida
```

### **Si Overfitting:**
```yaml
use_stc: true
fully_connected: false
edge_threshold: 0.4    # Más restrictivo
# Y/o aumenta dropout en model config
```

### **Para Comparar con EEGNet:**
```yaml
use_stc: false         # Mismos 24 canales que EEGNet
fully_connected: true  # Pero con estructura de grafo
```

---

## 🚀 **Ejemplos de Uso**

### **Regenerar dataset con nueva configuración:**
```bash
# Edita config.yaml primero
nano config/config.yaml

# Luego fuerza regeneración
python train.py --force-rebuild
```

### **Comparar configuraciones:**
```bash
# Setup 1: Fully connected
# Editar: fully_connected: true
python train.py --force-rebuild
cp checkpoints/best_model.pt checkpoints/fully_connected.pt

# Setup 2: Threshold
# Editar: fully_connected: false
python train.py --force-rebuild
cp checkpoints/best_model.pt checkpoints/threshold.pt

# Comparar resultados
cat output/test_results.json
```

---

## ❓ **FAQ**

**Q: ¿Qué es mejor, fully connected o threshold?**
A: Con RTX 4090, empezá con fully connected. Si ves overfitting, probá threshold.

**Q: ¿Source space o electrodos?**
A: Source space (use_stc: true) es más interpretable. Electrodos si querés comparar directamente con EEGNet.

**Q: ¿El parámetro `source_type` hace algo?**
A: No, lo eliminé. Solo importa `use_stc`.

**Q: ¿Puedo tener self-loops?**
A: Sí, edita: `self_loops: true` en config. Por defecto false.

**Q: ¿Cómo sé si mi threshold es bueno?**
A: Ejecutá `python analysis/analyze_graphs.py` y mira la densidad promedio de grafos.

---

## 🔍 **Debugging**

### **Verificar configuración actual:**
```python
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    
print("Nodos:", "100 parcelas" if config['data']['use_stc'] else "24 electrodos")
print("Grafo:", "Fully Connected" if config['data']['graph']['fully_connected'] else f"Threshold {config['data']['graph']['edge_threshold']}")
```

### **Ver estadísticas de grafos generados:**
```bash
python analysis/analyze_graphs.py
cat output/analysis/graph_features_all.csv | head
```

---

**Configuración actual en tu sistema:**
- ✅ Fully connected habilitado
- ✅ Source space (100 parcelas) por defecto
- ✅ Parámetro redundante `source_type` eliminado

