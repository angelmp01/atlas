# Resumen de Cambios Implementados

## 1. Corrección de `trips_total_length_km`

**Problema**: Se dividía por `n_trips_logistic` en lugar de `n_trips`.
- `n_trips_logistic = n_trips * 0.04` (estimación de vehículos de carga)
- `trips_total_length_km` = suma total de distancias de todos los viajes
- **Corrección**: Ahora se divide por `n_trips` para obtener distancia por viaje individual

**Archivos modificados**:
- `atlas_ml/io.py`: 
  - Añadido campo `n_trips` al ORM de `SoddLoad`
  - Actualizada consulta SQL para incluir `n_trips`
  - Corregida división: `od_length_km = trips_total_length_km / n_trips`
  - Agregado incluye ahora tanto `n_trips_total` como `n_trips_daily` (logistic)

---

## 2. Aumento de Capacidad del Modelo

**Problema**: Con 20M de registros y 386 localizaciones, el modelo con parámetros conservadores solo aprendía medias (underfitting).

**Solución implementada** (`atlas_ml/config.py`):

```python
# ANTES:
"max_depth": 6
"n_estimators": 100
"learning_rate": 0.1

# AHORA:
"max_depth": 10              # +67% profundidad → patrones más complejos
"n_estimators": 500          # 5x más árboles → mejor ajuste
"learning_rate": 0.05        # -50% → aprendizaje más estable
"subsample": 0.8             # Regularización contra overfitting
"colsample_bytree": 0.8      # Regularización contra overfitting
"min_child_weight": 1        # Regularización adicional
"max_bin": 256               # Mayor precisión en histogramas
```

### Impacto Esperado:

**Calidad de predicciones**:
- ✅ Captura de patrones más complejos (interacciones entre features)
- ✅ Menos bias (no se quedará en "predecir solo medias")
- ✅ Mayor variación en predicciones
- ⚠️ Riesgo controlado de overfitting (mitigado con subsample/colsample)

**Tiempo de entrenamiento**:
- **Estimación**: De 1-2 horas → **3-5 horas**
- Razón: 5x más árboles + mayor profundidad
- GPU (CUDA) ayudará a mantenerlo razonable
- La validación cruzada será la parte más lenta (múltiples folds)

**Recomendación**: Monitorizar las métricas de CV para ver si mejora. Si sigue sin variación:
1. Verificar que los features tienen variabilidad (no son constantes)
2. Considerar añadir features de interacción (origen×destino, día×origen, etc.)
3. Revisar la agregación diaria (quizás perder información temporal)

---

## 3. Eliminación de Conversión a Probabilidad por Minuto

**Cambio**: El modelo ahora devuelve directamente **viajes por día** en lugar de probabilidades por minuto.

**Antes**:
```python
# Predicción
daily_trips = model.predict(X)
# Conversión a probabilidad por minuto (asunción uniforme)
lambda_per_minute = daily_trips / 1440
probability = 1 - exp(-lambda_per_minute)
return probability  # Valor entre 0-1
```

**Ahora**:
```python
# Predicción directa
daily_trips = model.predict(X)
return daily_trips  # Número esperado de vehículos de carga por día
```

**Ventajas**:
- ✅ **Más interpretable**: "Se esperan 3.5 vehículos hoy" vs "0.0024 de probabilidad por minuto"
- ✅ **Sin asunciones**: No asume distribución uniforme (que era irreal)
- ✅ **Más útil**: Para el camionero es más claro el número esperado de viajes
- ✅ **Directo**: Coincide exactamente con lo que predice el modelo

**Estructura actualizada**:
```python
@dataclass
class CandidateOutput:
    i_location_id: int
    d_location_id: int
    tau_minutes: int
    expected_trips_per_day: float  # NUEVO: viajes esperados por día
    exp_price: float
    exp_weight: float
```

---

## 4. Visualización de Resultados de Validación

**Pregunta**: "¿Puedo ver los resultados de test?"

**Respuesta**: **SÍ**, el código ya los genera. Se encuentran en:

### Durante el entrenamiento:
Los logs muestran métricas de CV:
```
Processing fold 1/N
Processing fold 2/N
...
MAE: X.XXX, R²: 0.XXX
```

### En el modelo guardado:
El archivo `model_card.json` contiene todas las métricas:
```json
{
  "performance": {
    "cv_metrics": {
      "mae_mean": 2.345,
      "mae_std": 0.123,
      "r2_mean": 0.678,
      ...
    },
    "overall_metrics": {
      "mae": 2.345,
      "rmse": 3.456,
      "r2": 0.678
    },
    "fold_results": [...]
  }
}
```

### Para visualizar mejor:
Puedes añadir un script que lea los resultados:

```python
from atlas_ml import load_bundle

bundle = load_bundle("models/probability_v20241025_123456")
metrics = bundle.metadata["performance"]

print(f"MAE: {metrics['overall_metrics']['mae']:.3f}")
print(f"R²: {metrics['overall_metrics']['r2']:.3f}")
print(f"N folds: {bundle.metadata['training_config']['cv_setup']['n_folds']}")
```

---

## 5. Caso de Uso: Encontrar Cargas Intermedias

**Tu objetivo**: Dado un camionero que va de A → B, recomendar puntos intermedios X, Y, Z donde puede recoger cargas adicionales.

**Implementación sugerida**:

```python
from atlas_ml import load_bundle, CandidateInput
import datetime

# Cargar modelos
trip_model = load_bundle("models/probability_v...")
price_model = load_bundle("models/price_v...")

# Ruta del camionero
origen_A = 1  # Barcelona
destino_B = 50  # Madrid

# Puntos intermedios candidatos (ejemplo: Zaragoza, Tarragona, etc.)
puntos_intermedios = [15, 23, 42, 67, 89]  # Location IDs

# Fecha y contexto
hoy = datetime.date.today()
dia_semana = hoy.weekday()
semana_año = hoy.isocalendar().week

# Evaluar cada punto intermedio
candidatos = []
for punto_X in puntos_intermedios:
    # Carga de X → B (destino final)
    candidato = CandidateInput(
        i_location_id=punto_X,  # Origen intermedio
        d_location_id=destino_B,  # Destino final
        truck_type='normal',
        tipo_mercancia='normal',
        day_of_week=dia_semana,
        week_of_year=semana_año,
        holiday_flag=0,
        tau_minutes=0  # No usado actualmente
    )
    candidatos.append(candidato)

# Predecir viajes esperados
from atlas_ml import predict_probability
viajes_esperados = predict_probability(candidatos, trip_model)

# Predecir precios
from atlas_ml import predict_price
precios_esperados = predict_price(candidatos, price_model)

# Ranking de puntos intermedios
resultados = []
for i, punto_X in enumerate(puntos_intermedios):
    resultados.append({
        'punto': punto_X,
        'viajes_por_dia': viajes_esperados[i],
        'precio_esperado': precios_esperados[i],
        'score': viajes_esperados[i] * precios_esperados[i]  # Métrica combinada
    })

# Ordenar por score
resultados.sort(key=lambda x: x['score'], reverse=True)

# Recomendar top 3
print("Top 3 puntos intermedios recomendados:")
for r in resultados[:3]:
    print(f"  Punto {r['punto']}: {r['viajes_por_dia']:.2f} viajes/día, "
          f"€{r['precio_esperado']:.2f} precio esperado, "
          f"Score: {r['score']:.2f}")
```

**Output ejemplo**:
```
Top 3 puntos intermedios recomendados:
  Punto 23: 4.5 viajes/día, €350.00 precio esperado, Score: 1575.00
  Punto 15: 3.2 viajes/día, €280.00 precio esperado, Score: 896.00
  Punto 42: 2.8 viajes/día, €310.00 precio esperado, Score: 868.00
```

**Mejoras posibles**:
1. Filtrar solo puntos que estén en la ruta (usando geografía)
2. Considerar el tiempo de desvío (distancia extra)
3. Balance entre número de viajes y precio
4. Tiempo de espera estimado (si hay pocas cargas)

---

## Próximos Pasos Recomendados

1. **Re-entrenar con nuevos parámetros**:
   ```bash
   python train.py --only probability
   ```

2. **Monitorizar métricas de CV**:
   - Ver si MAE disminuye
   - Ver si R² aumenta
   - Comprobar que hay variación en predicciones

3. **Analizar features importantes**:
   - Usar `model.feature_importances_` de XGBoost
   - Identificar qué features aportan más

4. **Si sigue prediciendo medias**:
   - Añadir features de interacción
   - Revisar si faltan features importantes (hora del día, etc.)
   - Considerar modelos más complejos (LightGBM, CatBoost)

5. **Implementar API de recomendación**:
   - Crear endpoint que reciba A→B
   - Devolver top-N puntos intermedios
   - Integrar con sistema de routing
