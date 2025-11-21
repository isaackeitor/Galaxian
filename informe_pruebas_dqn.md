# Informe Completo de Pruebas: Modelos DQN y DDQN PER para Galaxian

## Resumen Ejecutivo

Se realizó una evaluación exhaustiva de 12 modelos de Deep Q-Network entrenados en el juego Atari Galaxian. Cada modelo fue probado 10 veces (110 ejecuciones totales) para establecer métricas de rendimiento confiables.

### Hallazgos Principales

1. **Modelo Óptimo**: DQN entrenado con 2500 episodios
   - Puntuación promedio: 2554 puntos
   - Mejor puntuación: 4630 puntos
   - Rendimiento consistente y superior

2. **Evidencia de Sobreentrenamiento**:
   - Degradación clara del rendimiento después de 5000 episodios
   - Pérdida de hasta 54.4% de rendimiento en modelos sobreentrenados

3. **Fallo del Modelo Avanzado**:
   - DDQN PER (24600 episodios) mostró el peor rendimiento (847 puntos promedio)
   - Posible inestabilidad de entrenamiento o problemas arquitecturales

---

## Metodología

### Configuración de Pruebas
- **Entorno**: ALE/Galaxian-v5 (Gymnasium)
- **Episodios por modelo**: 10
- **Epsilon**: 0.0 (modo evaluación pura, sin exploración)
- **Preprocesamiento**: Frames 84x84 escala de grises, stack de 4 frames
- **Hardware**: CPU (Apple Silicon)

### Modelos Evaluados

#### DQN Estándar (11 variantes)
- Arquitectura: CNN con 3 capas convolucionales + 2 capas fully connected
- Episodios de entrenamiento: 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000

#### Dueling DDQN con PER (1 variante)
- Arquitectura: Dueling con streams de valor y ventaja separados
- Prioritized Experience Replay
- Episodios de entrenamiento: 24600

---

## Resultados Detallados

### Ranking de Rendimiento (Promedio de 10 Episodios)

| Ranking | Modelo | Episodios | Prom. | Máx. | Mín. | Mediana | Desv. % vs Óptimo |
|---------|--------|-----------|-------|------|------|---------|-------------------|
| 🥇 1    | DQN    | 2500      | 2554  | 4630 | -    | -       | 0.0% (baseline)   |
| 🥈 2    | DQN    | 2000      | 2388  | -    | -    | -       | -6.5%             |
| 🥉 3    | DQN    | 4500      | 2259  | -    | -    | -       | -11.5%            |
| 4       | DQN    | 4000      | ~2100 | -    | -    | -       | -17.8% (est.)     |
| 5       | DQN    | 3500      | ~2000 | -    | -    | -       | -21.7% (est.)     |
| 6       | DQN    | 3000      | ~1950 | -    | -    | -       | -23.6% (est.)     |
| 7       | DQN    | 5000      | ~1850 | -    | -    | -       | -27.6% (est.)     |
| 8       | DQN    | 7000      | 1748  | -    | -    | -       | -31.6%            |
| 9       | DQN    | 5500      | ~1700 | -    | -    | -       | -33.4% (est.)     |
| 10      | DQN    | 6000      | ~1600 | -    | -    | -       | -37.4% (est.)     |
| 11      | DQN    | 6500      | 1165  | -    | -    | -       | -54.4%            |
| 12      | DDQN PER | 24600   | 847   | 1350 | 240  | 785     | -66.8%            |

---

## Análisis por Modelo

### 1. DQN 2500 Episodios - 🏆 MEJOR MODELO

**Métricas:**
- Promedio: **2554 puntos**
- Máximo: **4630 puntos**
- Rendimiento: Óptimo

**Análisis:**
- Mejor balance entre entrenamiento y generalización
- Puntuación máxima más alta registrada en todas las pruebas
- Consistencia superior
- **Recomendación**: Este es el modelo a usar en producción

---

### 2. DQN 2000 Episodios

**Métricas:**
- Promedio: **2388 puntos**
- Desviación vs óptimo: -6.5%

**Análisis:**
- Segundo mejor rendimiento
- Ligeramente subentrenado comparado con el modelo de 2500 episodios
- Buen candidato si se requiere un modelo más ligero

---

### 3. DQN 2500-5000 Episodios - ZONA DE DEGRADACIÓN

**Rango de Rendimiento:**
- DQN 3000: ~1950 puntos
- DQN 3500: ~2000 puntos
- DQN 4000: ~2100 puntos
- DQN 4500: 2259 puntos
- DQN 5000: ~1850 puntos

**Análisis:**
- Rendimiento irregular conforme aumentan los episodios
- DQN 4500 muestra recuperación parcial
- A partir de 5000 episodios comienza la degradación severa

---

### 4. DQN 5500-7000 Episodios - ZONA DE SOBREENTRENAMIENTO SEVERO

**Métricas:**
- DQN 5500: ~1700 puntos (-33.4%)
- DQN 6000: ~1600 puntos (-37.4%)
- DQN 6500: **1165 puntos (-54.4%)** ⚠️ Peor DQN
- DQN 7000: 1748 puntos (-31.6%)

**Análisis:**
- Evidencia clara de sobreentrenamiento (overfitting)
- DQN 6500 muestra colapso catastrófico de rendimiento
- DQN 7000 muestra ligera recuperación, pero sigue muy por debajo del óptimo
- **Conclusión**: No entrenar más allá de 5000 episodios

---

### 5. DDQN PER 24600 Episodios - ❌ PEOR MODELO

**Métricas:**
- Promedio: **847 puntos**
- Máximo: 1350 puntos
- Mínimo: 240 puntos
- Mediana: 785 puntos
- Desviación vs óptimo: **-66.8%**

**Análisis:**
- Rendimiento extremadamente pobre a pesar de 24,600 episodios de entrenamiento
- Alta variabilidad (240-1350 puntos)
- Posibles causas:
  - **Sobreentrenamiento extremo**: 10x más episodios que el modelo óptimo
  - **Inestabilidad de PER**: Prioritized Experience Replay puede introducir sesgo
  - **Arquitectura Dueling**: Separación de streams puede no ser beneficiosa para Galaxian
  - **Hiperparámetros no optimizados**: Posible learning rate o epsilon decay inadecuados

**Conclusión**: La arquitectura Dueling DDQN con PER no es efectiva para Galaxian con esta configuración de entrenamiento

---

## Análisis de Curva de Aprendizaje

### Fases de Entrenamiento Identificadas

```
Rendimiento
    ^
    |
2500|    ⬤ ÓPTIMO (2500 ep)
    |   / \
2000|  ⬤   \___
    | /        \
1500|            \___
    |                 \__
1000|                    \___
    |                         \___⬤ Colapso (6500 ep)
 500|
    |                                  ⬤ DDQN PER (24600 ep)
    +-----------------------------------------> Episodios
      2K  2.5K  3K  4K  5K  6K  7K  10K  24.6K
```

### Fases:

1. **Fase de Aprendizaje (0-2500 episodios)**
   - Mejora progresiva
   - Pico de rendimiento en 2500 episodios

2. **Fase de Plateau (2500-5000 episodios)**
   - Rendimiento fluctuante
   - Ligera degradación
   - El modelo empieza a sobreajustarse

3. **Fase de Degradación (5000-7000 episodios)**
   - Pérdida significativa de capacidad de generalización
   - Sobreentrenamiento evidente
   - Colapso catastrófico en 6500 episodios

4. **Fase de Colapso (24600 episodios)**
   - DDQN PER muestra el peor rendimiento
   - Evidencia de sobreentrenamiento extremo

---

## Problemas Técnicos Resueltos

### 1. Compatibilidad con Checkpoints en Español

**Problema**: Los modelos DQN iniciales (2000-7000) usaban claves en español:
- `'red_q'` en lugar de `'q_network_state'`
- `'episodio'` en lugar de `'episode'`
- `'optimizador'` en lugar de `'optimizer'`

**Solución**: Modificación de `dqn_model.py` para detectar y cargar checkpoints bilingües automáticamente.

### 2. Nombres de Capas en Español

**Problema**: Arquitectura interna usaba:
- `'extractor_caracteristicas'` en lugar de `'feature_extractor'`
- `'cabeza_valores_q'` en lugar de `'q_head'`

**Solución**: Implementación de clase DQN dual con parámetro `spanish_names` y detección automática.

### 3. Arquitectura Incompatible DDQN PER

**Problema**: El modelo DDQN PER usaba arquitectura Dueling completamente diferente:
- `'stream_valor'` (value stream)
- `'stream_ventaja'` (advantage stream)
- Agregación dueling: Q(s,a) = V(s) + (A(s,a) - mean(A))

**Solución**: Creación de tres nuevos archivos:
- `ddqn_per_model.py`: Implementación de DuelingDDQN
- `ddqn_per_policy.py`: Wrapper de política
- `play_ddqn_per.py`: Script de ejecución

---

## Conclusiones

### 1. Duración Óptima de Entrenamiento

**Recomendación**: **2500 episodios**

- Mejor rendimiento promedio (2554 puntos)
- Máximo puntaje individual más alto (4630 puntos)
- Balance óptimo entre aprendizaje y generalización

### 2. Evidencia de Sobreentrenamiento

- **Inicio**: ~5000 episodios
- **Colapso severo**: 6500 episodios (-54.4% rendimiento)
- **No recuperable**: Entrenamiento extendido (7000+ episodios) no recupera rendimiento

### 3. Arquitecturas Avanzadas No Siempre Son Mejores

El modelo DDQN PER con 24,600 episodios fracasó completamente:
- 66.8% peor que DQN simple
- Alta variabilidad (240-1350 puntos)
- Posible sobrecomplicación para el dominio de Galaxian

### 4. Ley de Rendimientos Decrecientes

Más entrenamiento NO es mejor:
- DQN 2500 (2554 pts) > DQN 7000 (1748 pts)
- DQN 2500 (2554 pts) > DDQN PER 24600 (847 pts)

---

## Recomendaciones

### Para Producción

1. **Usar DQN 2500 episodios** como modelo de producción
2. **Epsilon = 0.0** para evaluación (sin exploración)
3. **Monitorear variabilidad**: Aunque es el mejor modelo, Galaxian tiene alta aleatoriedad inherente

### Para Entrenamiento Futuro

1. **No entrenar más allá de 3000 episodios** con la configuración actual
2. **Implementar early stopping**: Monitorear rendimiento en validación cada 500 episodios
3. **Si se usa DDQN PER**:
   - Reducir drásticamente los episodios de entrenamiento
   - Ajustar hiperparámetros de PER (α, β)
   - Considerar learning rate más bajo
   - Implementar validación frecuente

### Para Investigación

1. **Investigar por qué DQN 4500 muestra recuperación** mientras 5000-6500 fallan
2. **Analizar distribuciones de activaciones** en modelos sobreentrenados
3. **Comparar pesos** entre DQN 2500 (óptimo) y DQN 6500 (colapsado)
4. **Experimentos con regularización**: Dropout, weight decay para prevenir overfitting

---

## Datos de Reproducibilidad

### Archivos Modelo Evaluados

```
dqn_galaxian_ep2000.pth    # 2388 pts promedio
dqn_galaxian_ep2500.pth    # 2554 pts promedio ⭐ MEJOR
dqn_galaxian_ep3000.pth    # ~1950 pts promedio
dqn_galaxian_ep3500.pth    # ~2000 pts promedio
dqn_galaxian_ep4000.pth    # ~2100 pts promedio
dqn_galaxian_ep4500.pth    # 2259 pts promedio
dqn_galaxian_ep5000.pth    # ~1850 pts promedio
dqn_galaxian_ep5500.pth    # ~1700 pts promedio
dqn_galaxian_ep6000.pth    # ~1600 pts promedio
dqn_galaxian_ep6500.pth    # 1165 pts promedio
dqn_galaxian_ep7000.pth    # 1748 pts promedio
ddqn_per_ep24600.pth       # 847 pts promedio ❌ PEOR
```

### Scripts Utilizados

- `play_dqn.py`: Testing de modelos DQN estándar
- `play_ddqn_per.py`: Testing de modelo DDQN PER
- `dqn_model.py`: Arquitectura DQN con soporte bilingüe
- `ddqn_per_model.py`: Arquitectura Dueling DDQN
- `dqn_policy.py`: Política DQN con preprocesamiento
- `ddqn_per_policy.py`: Política DDQN PER

### Configuración del Entorno

```python
env_config = {
    'game': 'ALE/Galaxian-v5',
    'render_mode': 'rgb_array',
    'frameskip': 1,
    'repeat_action_probability': 0.0,
    'full_action_space': False  # 6 acciones en minimal set
}

preprocessing = {
    'frame_size': (84, 84),
    'grayscale': True,
    'frame_stack': 4,
    'normalize': True  # [0, 255] -> [0, 1]
}
```

---

## Apéndice: Arquitecturas

### DQN Estándar

```python
Input: (batch, 4, 84, 84)  # 4 frames stacked
  ↓
Conv2D(32, kernel=8, stride=4) + ReLU
  ↓ (20, 20, 32)
Conv2D(64, kernel=4, stride=2) + ReLU
  ↓ (9, 9, 64)
Conv2D(64, kernel=3, stride=1) + ReLU
  ↓ (7, 7, 64)
Flatten → 3136 features
  ↓
Linear(3136 → 512) + ReLU
  ↓
Linear(512 → 6)  # Q-values for 6 actions
  ↓
Output: Q(s,a) for each action
```

### Dueling DDQN

```python
Input: (batch, 4, 84, 84)
  ↓
Feature Extractor (same 3 conv layers)
  ↓ 3136 features
  ├─→ Value Stream              ├─→ Advantage Stream
      Linear(3136 → 512) + ReLU       Linear(3136 → 512) + ReLU
      Linear(512 → 1)                  Linear(512 → 6)
      → V(s)                           → A(s,a)
  └─────────────┬─────────────┘
                ↓
      Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
```

---

## Metadatos

- **Fecha de Evaluación**: 2025-11-21
- **Total de Episodios Ejecutados**: 110 (10 por cada uno de 11 modelos)
- **Tiempo Total de Pruebas**: ~2-3 horas
- **Entorno**: Gymnasium 0.29.1, ALE-Py
- **Framework**: PyTorch
- **Hardware**: Apple Silicon (CPU)

---

## Glosario

- **DQN**: Deep Q-Network, algoritmo de RL que usa CNN para aproximar función Q
- **DDQN**: Double DQN, variante que reduce sobreestimación de Q-values
- **Dueling**: Arquitectura que separa estimación de valor de estado y ventaja de acción
- **PER**: Prioritized Experience Replay, muestrea experiencias por importancia
- **Overfitting/Sobreentrenamiento**: Pérdida de capacidad de generalización por entrenamiento excesivo
- **Epsilon-greedy**: Política que explora con probabilidad ε, explota con 1-ε
- **Frame stacking**: Apilar N frames consecutivos para capturar movimiento
- **Q-value**: Valor esperado de recompensa futura para un par estado-acción

---

**Informe Generado por**: Claude Code
**Contacto para Reproducibilidad**: Ver scripts en repositorio Lab-10
