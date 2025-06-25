"""
Streamlit frontend for the Variant RAG application (`main_variant_ollama.py`).

Features:
- Provides a user interface to ask questions to the variant backend.
- Uses aiocache with Redis for caching responses.
- Connects to the backend API endpoint specified by VARIANT_API_URL.

Environment Variables:
- VARIANT_API_URL: URL for the variant FastAPI backend (default: http://localhost:8001/ask).
- REDIS_URL: URL for Redis instance used by this Streamlit app's cache (default: redis://localhost:6379).
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
# This Redis cache is specific to this Streamlit app instance.
async def get_cache():
    # Consider making Redis URL configurable
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis = await aioredis.create_redis_pool(redis_url)
    cache = Cache(Cache.REDIS, endpoint=redis_url.split("://")[1].split(":")[0], port=int(redis_url.split(":")[-1]), namespace="streamlit_variant_cache", redis=redis, serializer=JsonSerializer())
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
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_data = response.json()
            await set_response_to_cache(prompt, response_data)
            return response_data
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
            return None
        except json.JSONDecodeError:
            st.error("Error decoding API response. Is the backend running and returning valid JSON?")
            return None


# Título de la aplicación
st.title("Health and Wellness Assistant (Variant)")

st.write("""
    Bienvenido al Asistente de Salud y Bienestar (Versión Variante).
    Este bot está diseñado para responder preguntas sobre salud y bienestar,
    utilizando múltiples fuentes de información incluyendo web scraping,
    documentos PDF cargados (PDF, CSV, JSON) y búsquedas en FAISS con OllamaEmbeddings.
    Puede ayudarte a entender mejor condiciones médicas, encontrar información
    sobre tratamientos y síntomas, y proporcionarte datos genéticos relacionados
    con diversas enfermedades.
""")

# API URL for the variant backend
# Defaulting to port 8001 for the variant to avoid conflict with main app on 8000
# This should be configured if running differently.
VARIANT_API_URL = os.getenv("VARIANT_API_URL", "http://localhost:8001/ask")
st.sidebar.info(f"Connecting to variant API at: {VARIANT_API_URL}")

# Entrada de texto para la pregunta del usuario
question = st.text_input("¿Tienes alguna pregunta sobre salud o genética? (Variant App)")

# Botón para obtener la respuesta
if st.button("Obtener respuesta (Variant)"):
    if question:
        response_data = asyncio.run(get_response_from_cache(question))
        if not response_data:
            response_data = asyncio.run(fetch_response(question, VARIANT_API_URL))

        if response_data:
            st.write("Respuesta:")
            # Ensure 'respuesta' key exists, matching main_variant_ollama.py
            if 'respuesta' in response_data:
                with st.spinner("Generando la respuesta..."):
                    st.write(response_data['respuesta'])
            else:
                st.error("La respuesta del API no contiene la clave 'respuesta'.")
                st.json(response_data) # Show full response for debugging

            # Ensure 'contexto' key exists for details
            if 'contexto' in response_data and response_data['contexto']:
                with st.expander("Detalles adicionales (Contexto)"):
                    for step in response_data["contexto"]:
                        st.write(step)
            else:
                st.info("No hay detalles de contexto adicionales proporcionados.")

            st.success("¡Respuesta generada con éxito!")
            st.balloons()
    else:
        st.warning("Por favor, introduce una pregunta.")
