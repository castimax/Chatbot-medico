import streamlit as st
import requests
import asyncio
from aiocache import Cache
from aiocache.serializers import JsonSerializer

# Initialize Cache object once globally.
# aiocache.Cache constructor is synchronous. Async operations are its methods.
# It will manage its own connection to Redis.
try:
    global_cache = Cache(
        Cache.REDIS,
        endpoint="localhost",
        port=6379,
        namespace="main",
        serializer=JsonSerializer()
    )
except Exception as e:
    st.error(f"Failed to initialize cache: {e}. Caching will be disabled.")
    global_cache = None

async def get_response_from_cache(prompt):
    if not global_cache:
        return None
    try:
        return await global_cache.get(prompt)
    except Exception as e:
        st.warning(f"Cache read error: {e}. Fetching from API.")
        return None

async def set_response_to_cache(prompt, response_data):
    if not global_cache:
        return
    try:
        await global_cache.set(prompt, response_data, ttl=60*60)  # Cachear por 1 hora
    except Exception as e:
        st.warning(f"Cache write error: {e}.")

async def fetch_response(prompt):
    with st.spinner("Procesando la pregunta..."):
        response = requests.post("http://localhost:8000/ask", json={"question": prompt})
        response_data = response.json()
    await set_response_to_cache(prompt, response_data)
    return response_data

# Título de la aplicación
st.title("Health and Wellness Assistant")

st.write("""
    Bienvenido al Asistente de Salud y Bienestar.
    Este bot está diseñado para responder preguntas sobre salud y bienestar,
    utilizando múltiples fuentes de información incluyendo web scraping,
    documentos PDF, búsquedas en FAISS y más.
    Puede ayudarte a entender mejor condiciones médicas, encontrar información
    sobre tratamientos y síntomas, y proporcionarte datos genéticos relacionados
    con diversas enfermedades.
""")

# Entrada de texto para la pregunta del usuario
question = st.text_input("¿Tienes alguna pregunta sobre salud o genética?")

# Botón para obtener la respuesta
if st.button("Obtener respuesta"):
    response_data = asyncio.run(get_response_from_cache(question))
    if not response_data:
        response_data = asyncio.run(fetch_response(question))
    
    st.write("Respuesta:")
    with st.spinner("Generando la respuesta..."):
        st.write(response_data.get('answer', "No se encontró respuesta.")) # Use .get for safety

    with st.expander("Detalles adicionales"):
        st.write("Contexto:")
        context_data = response_data.get('context', "No hay contexto adicional.") # Use .get for safety
        if isinstance(context_data, list):
            for step in context_data:
                st.write(step)
        else:
            st.write(context_data) # Display directly if it's not a list

    st.success("¡Respuesta generada con éxito!")
    st.balloons()
