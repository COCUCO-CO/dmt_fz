# Guía de Tipos de Convolución

## 🧠 Tipos de Convolución Soportados

Tu modelo ahora soporta **2 tipos de convolución**:

---

## 1️⃣ **GATv2Conv** (Graph Attention Networks v2)

```yaml
model:
  architecture:
    conv_type: "gatv2"
    num_attention_heads: 8
    attention_dropout: 0.2
    concat_heads: true
    use_edge_attr: true
```

### **Qué es:**
- Versión mejorada de GAT (Brody et al., 2021)
- Usa mecanismo de **attention** para ponderar vecinos
- Cada nodo "presta atención" a sus vecinos de forma diferente
- Puede usar **edge attributes** (valores de sincronización)

### **Cómo funciona:**
```
Para cada nodo i:
  1. Calcula attention weights α_ij para cada vecino j
  2. α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j || e_ij]))
  3. Agrega vecinos ponderados: h'_i = Σ_j α_ij · W·h_j
```

### **Ventajas:**
- ✅ **Interpretable:** Los pesos α_ij muestran qué conexiones son importantes
- ✅ **Adaptativo:** Aprende qué vecinos importan más
- ✅ **Edge attributes:** Puede usar valores de sincronización como features
- ✅ **Multi-head:** Captura diferentes tipos de relaciones

### **Desventajas:**
- ❌ Más parámetros (attention mechanism)
- ❌ Puede sobreajustar en datasets pequeños
- ❌ No usa estructura global del grafo (solo vecindario local)

### **Cuándo usar:**
- Querés interpretabilidad (visualizar attention)
- Tenés edge features importantes (sincronización)
- Dataset moderado/grande
- Querés aprender qué conexiones son relevantes

---

## 2️⃣ **ChebConv** (Chebyshev Convolution)

```yaml
model:
  architecture:
    conv_type: "cheby"
    cheby_k: 3  # Orden del filtro
```

### **Qué es:**
- Convolución **espectral** basada en polinomios de Chebyshev
- Usa el **Laplaciano del grafo** (estructura global)
- Aproximación eficiente de convoluciones espectrales
- Inspirado en CNNs pero para grafos

### **Cómo funciona:**
```
Para cada nodo i:
  1. Calcula Laplaciano normalizado L del grafo
  2. Usa polinomios Chebyshev T_k(L)
  3. h'_i = Σ_k θ_k · T_k(L̃) · h
  
K controla el "receptive field":
  - K=1: Solo vecinos inmediatos
  - K=3: Hasta 3-hop neighbors
  - K más alto = mayor contexto global
```

### **Ventajas:**
- ✅ **Usa estructura global:** Considera Laplaciano completo
- ✅ **Menos parámetros:** No tiene attention mechanism
- ✅ **Base teórica sólida:** Teoría de grafos espectral
- ✅ **Eficiente:** Computacionalmente más rápido que GAT
- ✅ **Regularizado:** Menos prone a overfitting

### **Desventajas:**
- ❌ **No usa edge attributes:** Ignora valores de sincronización
- ❌ Menos interpretable (no hay attention weights)
- ❌ K es hiperparámetro adicional
- ❌ Asume grafo fijo (no adaptativo)

### **Cuándo usar:**
- Dataset pequeño (menos overfitting)
- No te importa tanto la interpretabilidad
- Querés usar estructura global del grafo
- Entrenamiento más rápido
- Edge attributes no son tan informativos

---

## 🔬 **Comparación Lado a Lado**

| Aspecto | GATv2Conv | ChebConv |
|---------|-----------|----------|
| **Paradigma** | Spatial (attention) | Spectral (Laplaciano) |
| **Interpretabilidad** | Alta (α_ij) | Baja |
| **Edge attributes** | ✅ Sí | ❌ No |
| **Parámetros** | Más (attention) | Menos |
| **Overfitting risk** | Mayor | Menor |
| **Contexto** | Local (k-hop) | Global (Laplaciano) |
| **Velocidad** | Más lento | Más rápido |
| **Teoría** | Aprendizaje adaptativo | Teoría espectral |

---

## 🎯 **Configuraciones Recomendadas**

### **Para Exploración Inicial - GATv2:**
```yaml
model:
  architecture:
    conv_type: "gatv2"
    num_gat_layers: 3
    hidden_dim: 64
    num_attention_heads: 4
    attention_dropout: 0.3
    dropout: 0.5
    use_edge_attr: true
    concat_heads: true
```

**Por qué:**
- Usa tus valores de sincronización (edge attributes)
- Interpretable (puedes ver attention)
- Aprende adaptativamente

### **Para Baseline Rápido - Chebyshev:**
```yaml
model:
  architecture:
    conv_type: "cheby"
    num_gat_layers: 3
    hidden_dim: 64
    cheby_k: 3
    dropout: 0.5
```

**Por qué:**
- Menos parámetros (menos overfitting)
- Más rápido de entrenar
- Usa estructura global del grafo

---

## 🔧 **Cómo Cambiar**

### **Usar GATv2 (con attention y edge attributes):**
```bash
nano config/config.yaml
```

```yaml
model:
  architecture:
    conv_type: "gatv2"  # ← Cambiar aquí
```

### **Usar Chebyshev (espectral, sin edge attributes):**
```yaml
model:
  architecture:
    conv_type: "cheby"  # ← Cambiar aquí
    cheby_k: 3          # Orden del filtro
```

Luego:
```bash
python train.py --force-rebuild
```

---

## 📊 **Experimento Recomendado**

Probá ambos y compará:

```bash
# 1. GATv2
# Editar: conv_type: "gatv2"
python train.py --force-rebuild
cp checkpoints/best_model.pt checkpoints/gatv2_model.pt
cp output/test_results.json output/gatv2_results.json

# 2. Chebyshev
# Editar: conv_type: "cheby"
python train.py --force-rebuild
cp checkpoints/best_model.pt checkpoints/cheby_model.pt
cp output/test_results.json output/cheby_results.json

# 3. Comparar
echo "GATv2 Accuracy:"
cat output/gatv2_results.json | grep test_accuracy
echo "Cheby Accuracy:"
cat output/cheby_results.json | grep test_accuracy
```

---

## 💡 **Mi Recomendación para Tu Caso**

**Con 24 electrodos + fully connected:**

### **Opción 1: GATv2 (Recomendada)**
```yaml
use_stc: false           # 24 electrodos
fully_connected: true    # Todas las conexiones
conv_type: "gatv2"       # Usa valores de sincronización
```

**Por qué:**
- Tus valores de sincronización entre electrodos son informativos
- GATv2 puede aprenderlas vía edge_attr
- Attention te muestra qué pares de electrodos son importantes
- Puedes relacionar con tu análisis de Pearson

### **Opción 2: Chebyshev (Alternativa)**
```yaml
use_stc: false
fully_connected: true
conv_type: "cheby"
cheby_k: 3
```

**Por qué:**
- Dataset pequeño (29 sujetos)
- Chebyshev menos prone a overfitting
- Más rápido para iterar

---

## 📚 **Referencias**

**GATv2:**
- Brody et al. (2021). "How Attentive are Graph Attention Networks?"
- Mejora sobre GAT original (Veličković et al., 2018)

**Chebyshev:**
- Defferrard et al. (2016). "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
- Usa teoría espectral de grafos

---

## ❓ **FAQ**

**Q: ¿Puedo combinar ambos?**
A: No directamente, pero podés entrenar 2 modelos y hacer ensemble.

**Q: ¿Cuál es mejor?**
A: Depende de tus datos. GATv2 si edge attributes importantes, Cheby si dataset pequeño.

**Q: ¿Attention weights son necesarios?**
A: No necesarios, pero útiles para interpretabilidad y paper.

**Q: ¿K=3 es óptimo para Chebyshev?**
A: Es un buen default. Probá K=2 (más local) o K=4 (más global).

**Q: ¿Puedo usar edge attributes con Chebyshev?**
A: No, Chebyshev no soporta edge attributes. Usá GATv2 para eso.

---

**Configuración actual en tu sistema:**
- ✅ 24 electrodos (`use_stc: false`)
- ✅ Fully connected (`fully_connected: true`)
- ✅ GATv2 por defecto (`conv_type: "gatv2"`)

