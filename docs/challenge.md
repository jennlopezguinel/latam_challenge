# Resumen del Desarrollo del Modelo

## 1. Objetivos
- Convertir el contenido de `exploration.ipynb` en `model.py`.
- Garantizar las mejores prácticas para el desarrollo de Machine Learning basado en Python.
- Evaluar el rendimiento del modelo utilizando métricas de AUC y precisión.

## 2. Pasos Realizados
### Exploración de Datos
- Se agregó una función `exploratory_analysis` para visualizar:
  - Vuelos por aerolínea, día y mes.

### Preprocesamiento de Datos
- Se convirtieron las columnas de fecha (`Fecha-I`, `Fecha-O`) al formato datetime.
- Se manejaron valores nulos eliminando filas donde faltaban fechas.
- Se crearon nuevas características:
  - `period_day`: Categoriza los vuelos en mañana, tarde o noche.
  - `high_season`: Indica si el vuelo ocurrió durante una temporada de alta demanda.
  - `min_diff`: Diferencia en minutos entre los horarios programados y reales.
  - `delay`: Etiqueta binaria basada en un umbral de 15 minutos para los retrasos.

### Desarrollo del Modelo
- Se utilizó `LogisticRegression` para la clasificación binaria.
- Los datos se dividieron en conjuntos de entrenamiento y prueba (67% entrenamiento, 33% prueba).
- El modelo fue entrenado con `max_iter=1000` para garantizar la convergencia.

## 3. Resultados
- **AUC**: 0.517
- **Precisión**: 81.45%

## 4. Observaciones
- La alta precisión refleja un desequilibrio en la distribución de los retrasos (probablemente dominada por vuelos no retrasados).
- El AUC indica una baja capacidad para distinguir entre vuelos retrasados y no retrasados.
