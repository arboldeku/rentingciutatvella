# 🏘️ Predicción del precio de alquiler en Ciutat Vella (Barcelona)

Este proyecto aborda la estimación del precio mensual de alquiler en la ciudad de Barcelona, con un enfoque específico en el distrito de **Ciutat Vella**. Partiendo de un análisis de todos los distritos de la ciudad, se construyó un modelo de regresión ajustado a este distrito central para obtener predicciones precisas y accionables.

## 🔎 En qué consiste el proyecto

- Procesamiento y segmentación de datos de alquiler por distrito.
- Construcción de modelos de regresión múltiple, Ridge y Lasso.
- Evaluación del rendimiento predictivo y ajuste fino del modelo.
- Implementación de un sistema interactivo para introducir características y obtener una estimación de precio.

## 🧪 Enfoque metodológico

1. **Filtrado geográfico**: agrupación inicial por distritos → selección final de Ciutat Vella.
2. **Modelado**: 
   - Regresión lineal múltiple
   - Ridge y Lasso (penalizaciones para evitar overfitting)
   - Estandarización de variables para garantizar interpretabilidad
3. **Evaluación**: modelos comparados por R², error medio y consistencia
4. **Aplicación práctica**: creación de un modelo exportado (`.pkl`) que permite estimar precios con nuevos datos introducidos manualmente.

## 🧰 Herramientas y tecnologías

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib, Seaborn
- Pickle para exportación del modelo
- Interfaz básica en código para predicciones custom

## 📈 Resultados destacados

- Puedes encontrar todos los hallazgos, insights y resultados en mi sitio web: https://albertbaneresdata.softr.app/ciutatvellarent

## 💡 Posibles mejoras

- Crear una interfaz gráfica con `Streamlit` para facilitar la interacción del usuario final.
- Incorporar más variables como año de construcción o nivel de reforma del inmueble.
- Añadir validación cruzada para una evaluación más sólida del modelo.

  ✍️ Autor
  
[Albert Bañeres] - [https://www.linkedin.com/in/albert-ba%C3%B1eres-873a28137/]
