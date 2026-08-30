# Detección y clasificación automática de ciberacoso en redes sociales mediante Machine Learning

Este proyecto se ha desarrollado como Trabajo Fin de Máster del Máster en Data Science e Inteligencia Artificial de Nodd3r.

El objetivo es analizar mensajes publicados en redes sociales y clasificarlos automáticamente según diferentes tipos de ciberacoso mediante técnicas de Procesamiento del Lenguaje Natural (NLP) y Machine Learning.

El proyecto parte de un conjunto de mensajes de Twitter clasificados en seis categorías:

- Ciberacoso relacionado con la edad (`age`)
- Ciberacoso relacionado con la etnia (`ethnicity`)
- Ciberacoso relacionado con el género (`gender`)
- Ciberacoso relacionado con la religión (`religion`)
- Otros tipos de ciberacoso (`other_cyberbullying`)
- No ciberacoso (`not_cyberbullying`)

Además del análisis de los datos y el entrenamiento de diferentes modelos, se ha desarrollado una aplicación web con Streamlit que permite introducir un mensaje y obtener la clasificación realizada por el modelo.

## Aplicación

La aplicación se encuentra desplegada en Streamlit Community Cloud:

https://eva10lm-tfm-ciberacoso-app-flexfp.streamlit.app
## Dataset

Para el desarrollo del proyecto se ha utilizado el dataset **Cyberbullying Classification**, disponible en Kaggle.

El conjunto de datos original contiene **47.692 mensajes** y dos variables:

- `tweet_text`: texto del mensaje.
- `cyberbullying_type`: categoría asignada al mensaje.

Durante el análisis inicial de calidad de los datos se comprobó que no existían valores nulos, pero sí se encontraron textos duplicados y mensajes asociados a más de una categoría.

En concreto, se detectaron:

- 36 filas completamente duplicadas.
- 1.675 textos duplicados.
- 1.639 textos asociados a más de una etiqueta.

Para evitar introducir información contradictoria en el entrenamiento, se eliminaron los registros correspondientes a textos con etiquetas diferentes y posteriormente se eliminaron los textos duplicados restantes.

Después del proceso de limpieza, el dataset utilizado para el análisis y modelado quedó formado por **44.378 mensajes**, sin valores nulos, duplicados ni conflictos entre etiquetas.

La distribución final de las seis categorías se encuentra relativamente equilibrada, con porcentajes aproximados entre el 14 % y el 18 %, por lo que no fue necesario aplicar técnicas adicionales de balanceo.
## Metodología y modelos

El problema se planteó como una clasificación multiclase, ya que el objetivo es asignar cada mensaje a una de las seis categorías disponibles.

Para transformar los textos en información que pudiera ser utilizada por los modelos de Machine Learning se utilizó **TF-IDF (Term Frequency-Inverse Document Frequency)**.

La evaluación se realizó separando los datos en un conjunto de entrenamiento del 80 % y un conjunto de test del 20 %, utilizando una división estratificada para mantener una distribución similar de las clases en ambos conjuntos.

Se probaron inicialmente tres modelos de Machine Learning:

- Regresión Logística.
- Multinomial Naive Bayes.
- Linear SVM (`LinearSVC`).

Como métricas principales se utilizaron **Accuracy y Macro F1**, dando especial importancia a Macro F1 porque permite evaluar el rendimiento del modelo teniendo en cuenta todas las clases por igual.

Después de comparar los modelos iniciales también se realizaron experimentos incorporando unigramas y bigramas en la representación TF-IDF.

La **Regresión Logística** obtuvo los mejores resultados globales y fue el modelo seleccionado para la fase de optimización.

Para optimizarlo se utilizó `GridSearchCV` con validación cruzada sobre los datos de entrenamiento, evaluando diferentes valores del parámetro `C` y distintas configuraciones de n-gramas.

La mejor configuración obtenida fue:

- Modelo: Regresión Logística.
- `C = 2`.
- TF-IDF con unigramas y bigramas `(1,2)`.
- `min_df = 5`.
- `max_features = 30000`.

Todo el procesamiento del texto y el modelo se integraron mediante un `Pipeline` de scikit-learn, lo que permite aplicar de forma conjunta la vectorización TF-IDF y la clasificación.
## Resultados

El modelo final se evaluó sobre el conjunto de test, formado por datos que no habían sido utilizados durante el entrenamiento.

Los resultados globales obtenidos fueron:

- **Accuracy:** 0,8767
- **Macro F1:** 0,8653
- **Weighted F1:** 0,8771

El rendimiento no fue igual en todas las categorías. Los valores F1 obtenidos por clase fueron aproximadamente:

| Categoría | F1-score |
|---|---:|
| Edad (`age`) | 0,9644 |
| Etnia (`ethnicity`) | 0,9744 |
| Género (`gender`) | 0,8943 |
| Religión (`religion`) | 0,9563 |
| Otros tipos de ciberacoso (`other_cyberbullying`) | 0,7317 |
| No ciberacoso (`not_cyberbullying`) | 0,6706 |

Los mejores resultados se obtuvieron en las categorías relacionadas con edad, etnia y religión, mientras que las principales dificultades aparecen en las categorías `not_cyberbullying` y `other_cyberbullying`.

El análisis de la matriz de confusión muestra que una parte importante de los errores se produce precisamente entre estas dos categorías. Esto puede estar relacionado con que contienen mensajes más generales y heterogéneos, mientras que otras categorías presentan patrones de vocabulario más específicos.

Una vez realizada la evaluación, el modelo final se volvió a entrenar utilizando los **44.378 registros disponibles** y se guardó en el archivo `modelo_ciberacoso.pkl` para utilizarlo posteriormente en la aplicación Streamlit.
## Aplicación Streamlit

Como parte final del proyecto se ha desarrollado una aplicación web utilizando Streamlit.

La aplicación permite introducir un mensaje de texto y obtener la categoría predicha por el modelo. Para realizar la predicción se carga directamente el Pipeline final guardado en `modelo_ciberacoso.pkl`, que incluye tanto la transformación TF-IDF como el modelo de Regresión Logística.

La interfaz muestra las categorías con una descripción en español para facilitar su interpretación.

La aplicación está disponible en:

https://eva10lm-tfm-ciberacoso-app-flexfp.streamlit.app


## Limitaciones y posibles mejoras

Aunque los resultados obtenidos son buenos a nivel global, el proyecto presenta algunas limitaciones que deben tenerse en cuenta.

Una de las principales es la dificultad del modelo para diferenciar entre las categorías `not_cyberbullying` y `other_cyberbullying`, que son precisamente las clases con menor F1-score y donde se concentra una parte importante de los errores.

Además, el dataset contiene principalmente mensajes en inglés y procede de Twitter, por lo que no se ha validado el rendimiento del modelo con mensajes en español ni con textos procedentes de otras redes sociales.

También se observaron durante el análisis algunas palabras y expresiones muy específicas del propio dataset. Esto puede hacer que el modelo aprenda determinados patrones asociados al conjunto de entrenamiento que no necesariamente se comporten igual con datos nuevos procedentes de otros contextos.

Como posibles mejoras futuras se podrían plantear:

- Ampliar el conjunto de datos con mensajes procedentes de diferentes redes sociales.
- Incorporar más ejemplos en español y otros idiomas.
- Analizar con mayor profundidad las categorías `not_cyberbullying` y `other_cyberbullying`.
- Probar otras técnicas de procesamiento y representación del texto.
- Evaluar el modelo con un conjunto de datos externo para estudiar mejor su capacidad de generalización.

La aplicación debe entenderse como un prototipo de Machine Learning para el análisis automático de patrones textuales. La clasificación obtenida no sustituye una valoración profesional ni determina por sí sola la existencia de una situación real de ciberacoso.
## Estructura del repositorio

El repositorio contiene los principales archivos utilizados para el desarrollo y despliegue del proyecto:

```text
TFM_Ciberacoso/
│
├── notebooks/
│   ├── 01_EDA_TFM_Ciberacoso.ipynb
│   ├── 02_ML_Experiments.ipynb
│   └── 03_Streamlit_TFM_Ciberacoso.ipynb
│
├── app.py
├── modelo_ciberacoso.pkl
├── requirements.txt
└── README.md