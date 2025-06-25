"""
Streamlit frontend for the Main Health and Wellness Assistant application (`main.py`).

Features:
- Provides a user interface to ask questions to the main backend.
- Uses aiocache with Redis for caching responses.
- Connects to the backend API endpoint specified by MAIN_API_URL.
- Handles different context types returned by the backend (list for agent steps,
  string for custom context).

Environment Variables:
- MAIN_API_URL: URL for the main FastAPI backend (default: http://localhost:8000/ask).
- STREAMLIT_REDIS_URL: URL for Redis instance used by this Streamlit app's cache (default: redis://localhost:6379).
"""
import streamlit as st
import requests
import aioredis
import asyncio
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import os # Added import
import json # Added import

# Configuración inicial de aiocache para usar Redis
async def get_cache():
    # Consider making Redis URL configurable, e.g., os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_url = os.getenv("STREAMLIT_REDIS_URL", "redis://localhost:6379")
    redis_endpoint = redis_url.split("://")[1].split(":")[0]
    redis_port = int(redis_url.split(":")[-1])
    redis = await aioredis.create_redis_pool(redis_url)
    # Namespace "main_cache" to distinguish from variant app's cache if using same Redis
    cache = Cache(Cache.REDIS, endpoint=redis_endpoint, port=redis_port, namespace="main_cache", redis=redis, serializer=JsonSerializer())
    return cache

async def get_response_from_cache(prompt: str):
    cache = await get_cache()
    return await cache.get(prompt)

async def set_response_to_cache(prompt: str, response_data: dict):
    cache = await get_cache()
    await cache.set(prompt, response_data, ttl=60*60)  # Cachear por 1 hora

async def fetch_response(prompt: str, api_url: str):
    with st.spinner("Procesando la pregunta..."):
        try:
            response = requests.post(api_url, json={"question": prompt})
            response.raise_for_status()
            response_data = response.json()
            await set_response_to_cache(prompt, response_data)
            return response_data
        except requests.exceptions.RequestException as e:
            st.error(f"Error conectando al API: {e}")
            return None
        except json.JSONDecodeError:
            st.error("Error decodificando respuesta del API. ¿Está el backend corriendo y devolviendo JSON válido?")
            return None

# Título de la aplicación
st.title("Health and Wellness Assistant (Main)")

st.write("""
    Bienvenido al Asistente de Salud y Bienestar.
    Este bot está diseñado para responder preguntas sobre salud y bienestar,
    utilizando múltiples fuentes de información incluyendo web scraping,
    documentos PDF, búsquedas en Elasticsearch y más.
    Puede ayudarte a entender mejor condiciones médicas, encontrar información
    sobre tratamientos y síntomas, y proporcionarte datos genéticos relacionados
    con diversas enfermedades.
""")

# API URL for the main backend
MAIN_API_URL = os.getenv("MAIN_API_URL", "http://localhost:8000/ask")
st.sidebar.info(f"Connecting to main API at: {MAIN_API_URL}")

# Entrada de texto para la pregunta del usuario
question = st.text_input("¿Tienes alguna pregunta sobre salud o genética? (Main App)")

# Botón para obtener la respuesta
if st.button("Obtener respuesta (Main)"):
    if question:
        response_data = asyncio.run(get_response_from_cache(question))
        if not response_data:
            response_data = asyncio.run(fetch_response(question, MAIN_API_URL))

        if response_data:
            st.write("Respuesta:")
            if 'answer' in response_data:
                with st.spinner("Generando la respuesta..."):
                    st.write(response_data['answer'])
            else:
                st.error("La respuesta del API no contiene la clave 'answer'.")
                st.json(response_data)

            if 'processing_time' in response_data:
                st.write(f"Tiempo de procesamiento: {response_data['processing_time']:.2f} segundos")

            if 'context' in response_data and response_data['context']:
                with st.expander("Detalles adicionales (Contexto)"):
                    context_data = response_data["context"]
                    if isinstance(context_data, list): # Agent intermediate steps
                        for step in context_data:
                            st.write(step)
                    elif isinstance(context_data, str): # Custom string context
                        st.text(context_data)
                    else:
                        st.write(context_data) # Fallback for other types
            else:
                st.info("No hay detalles de contexto adicionales proporcionados.")

            st.success("¡Respuesta generada con éxito!")
            st.balloons()
    else:
        st.warning("Por favor, introduce una pregunta.")
