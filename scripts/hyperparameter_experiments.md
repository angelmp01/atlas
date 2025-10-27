# Guía de Experimentación con Hiperparámetros XGBoost

Esta guía muestra cómo usar los nuevos argumentos CLI para experimentar con diferentes configuraciones de XGBoost y aumentar la capacidad del modelo.

## 🎯 Objetivo: Aumentar Capacidad del Modelo

Los modelos actuales (~30 MB) pueden tener capacidad limitada para capturar toda la entropía del dataset. Podemos aumentar la capacidad mediante:

1. **Más árboles** (`--n-estimators`)
2. **Árboles más profundos** (`--max-depth`)
3. **Mayor precisión del histograma** (`--max-bin`)

## 📊 Experimentos Recomendados

### Experimento 1: Baseline (Ver Parámetros Actuales)
```bash
python train.py --show-xgb-params
```

### Experimento 2: Aumentar Capacidad Moderadamente
**Objetivo**: Modelo ~50-60 MB

```bash
python train.py --max-depth 12 --n-estimators 800 --max-bin 384
```

**Cambios**:
- `max_depth`: 10 → 12 (+20% profundidad)
- `n_estimators`: 500 → 800 (+60% árboles)
- `max_bin`: 256 → 384 (+50% precisión)

### Experimento 3: Aumentar Capacidad Agresivamente
**Objetivo**: Modelo ~100-120 MB

```bash
python train.py --max-depth 15 --n-estimators 1500 --max-bin 512
```

**Cambios**:
- `max_depth`: 10 → 15 (+50% profundidad)
- `n_estimators`: 500 → 1500 (+200% árboles)
- `max_bin`: 256 → 512 (+100% precisión)

### Experimento 4: Modelo de Alta Capacidad
**Objetivo**: Modelo ~200+ MB (máxima capacidad)

```bash
python train.py --max-depth 20 --n-estimators 2500 --max-bin 1024
```

**Cambios**:
- `max_depth`: 10 → 20 (+100% profundidad)
- `n_estimators`: 500 → 2500 (+400% árboles)
- `max_bin`: 256 → 1024 (+300% precisión)

**⚠️ ADVERTENCIA**: Entrenamiento muy lento (~2-3 horas)

### Experimento 5: Learning Rate Bajo + Muchos Árboles
**Objetivo**: Aprendizaje más fino y estable

```bash
python train.py --learning-rate 0.02 --n-estimators 2000 --max-depth 12
```

**Ventajas**:
- Aprendizaje más gradual
- Mejor generalización
- Menos overfitting

## 🎮 Quick Test para Validación Rápida

Antes de entrenar el dataset completo, valida configuraciones con `--quick-test`:

```bash
# Test rápido con alta capacidad
python train.py --quick-test --only probability --max-depth 15 --n-estimators 1000

# Test rápido con regularización
python train.py --quick-test --only probability --subsample 0.6 --colsample-bytree 0.6
```

## 🔍 Análisis de Resultados en Wandb

Todos los experimentos se loguean automáticamente a Wandb con:

1. **Tags**:
   - `custom-hyperparams`: Indica que usaste parámetros CLI
   - Ver tags en wandb para filtrar experimentos

2. **Config**:
   - Todos los parámetros XGBoost
   - `cli_overrides`: Qué parámetros cambiaste

3. **Métricas**:
   - Compara MAE, RMSE, R² entre configuraciones
   - Tamaño del modelo en MB
   - Tiempo de entrenamiento

## 📈 Estrategia de Búsqueda

### Fase 1: Búsqueda Gruesa (Quick Test)
```bash
# Probar 3-4 configuraciones diferentes con quick-test
python train.py --quick-test --only probability --max-depth 12 --n-estimators 800
python train.py --quick-test --only probability --max-depth 15 --n-estimators 1000
python train.py --quick-test --only probability --max-depth 18 --n-estimators 1500
```

### Fase 2: Refinamiento (Dataset Completo)
```bash
# Entrenar las 1-2 mejores configuraciones en dataset completo
python train.py --max-depth 15 --n-estimators 1200 --max-bin 512
```

### Fase 3: Fine-tuning
```bash
# Ajustar learning rate y regularización
python train.py --max-depth 15 --n-estimators 1500 --learning-rate 0.03 --subsample 0.75
```

## 🎯 Combinaciones Recomendadas

### Para Evitar Overfitting
```bash
python train.py --max-depth 12 --min-child-weight 3 --subsample 0.7 --colsample-bytree 0.7
```

### Para Máxima Capacidad
```bash
python train.py --max-depth 18 --n-estimators 2000 --max-bin 768 --learning-rate 0.03
```

### Balanceado (Recomendado para empezar)
```bash
python train.py --max-depth 13 --n-estimators 1000 --max-bin 384 --learning-rate 0.04
```

## 📝 Notas

- **Tiempo de entrenamiento**: Aumenta ~linealmente con `n_estimators`, ~exponencialmente con `max_depth`
- **Memoria GPU**: Modelos más grandes requieren más VRAM
- **Tamaño del modelo**: Aproximadamente proporcional a `n_estimators * max_depth`
- **Wandb Dashboard**: https://wandb.ai/pgaitl/atlas-ml

## ⚡ Quick Reference

| Parámetro | Default | Rango | Efecto en Capacidad | Efecto en Tiempo |
|-----------|---------|-------|---------------------|------------------|
| `max_depth` | 10 | 1-50 | ⬆️⬆️⬆️ Alto | ⬆️⬆️ Exponencial |
| `n_estimators` | 500 | 10-10000 | ⬆️⬆️⬆️ Alto | ⬆️ Lineal |
| `learning_rate` | 0.05 | 0.001-0.3 | ➡️ Neutral | ➡️ Neutral |
| `max_bin` | 256 | 16-1024 | ⬆️ Medio | ⬆️ Bajo |
| `subsample` | 0.8 | 0.1-1.0 | ⬇️ Baja | ⬇️ Baja |
| `colsample_bytree` | 0.8 | 0.1-1.0 | ⬇️ Baja | ⬇️ Baja |
| `min_child_weight` | 1 | 0-100 | ⬇️ Baja | ➡️ Neutral |
