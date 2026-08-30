
import streamlit as st
import joblib

# ----------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------------

st.set_page_config(
    page_title="Clasificación de ciberacoso",
    page_icon="💬",
    layout="centered"
)

# ----------------------------------------------------------
# CARGA DEL MODELO
# ----------------------------------------------------------

modelo = joblib.load("modelo_ciberacoso.pkl")

# ----------------------------------------------------------
# TRADUCCIÓN DE LAS CATEGORÍAS
# ----------------------------------------------------------

categorias_es = {
    "age": "Ciberacoso relacionado con la edad",
    "ethnicity": "Ciberacoso relacionado con la etnia",
    "gender": "Ciberacoso relacionado con el género",
    "religion": "Ciberacoso relacionado con la religión",
    "other_cyberbullying": "Otros tipos de ciberacoso",
    "not_cyberbullying": "No ciberacoso"
}

# ----------------------------------------------------------
# TÍTULO Y DESCRIPCIÓN
# ----------------------------------------------------------

st.title("💬 Clasificación automática de ciberacoso")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning
    para clasificar mensajes de redes sociales en diferentes
    categorías relacionadas con el ciberacoso.
    """
)

# ----------------------------------------------------------
# ENTRADA DEL USUARIO
# ----------------------------------------------------------

texto_usuario = st.text_area(
    "Introduce un mensaje para analizar:",
    height=150,
    placeholder="Escribe aquí el mensaje..."
)

# ----------------------------------------------------------
# BOTÓN DE PREDICCIÓN
# ----------------------------------------------------------

if st.button("Analizar mensaje"):

    # Comprobamos que se haya introducido algún texto.
    if texto_usuario.strip() == "":

        st.warning(
            "Introduce un mensaje antes de realizar el análisis."
        )

    else:

        # El Pipeline recibe directamente el mensaje.
        # Internamente aplica TF-IDF y posteriormente
        # la Regresión Logística.
        prediccion = modelo.predict(
            [texto_usuario]
        )[0]

        # Traducimos la etiqueta original al español.
        categoria_mostrada = categorias_es.get(
            prediccion,
            prediccion
        )

        # Mostramos la predicción.
        st.success(
            f"Categoría detectada: {categoria_mostrada}"
        )

# ----------------------------------------------------------
# INFORMACIÓN DEL MODELO
# ----------------------------------------------------------

st.divider()

st.caption(
    "Modelo: Regresión Logística + TF-IDF "
    "(unigramas y bigramas)"
)

# ----------------------------------------------------------
# ADVERTENCIA
# ----------------------------------------------------------

st.info(
    """
    Esta aplicación es un prototipo de Machine Learning
    destinado al análisis automático de patrones textuales.

    La predicción no constituye una valoración profesional
    ni determina por sí sola la existencia de una situación
    real de ciberacoso.
    """
)
