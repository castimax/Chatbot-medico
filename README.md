# Chatbot con FastAPI, Streamlit y LangChain (Versiones V1 y V2)

Este repositorio contiene dos versiones coexistentes de un chatbot desarrollado utilizando FastAPI, Streamlit y LangChain. Ambas versiones utilizan modelos de lenguaje de `ChatGroq`.

*   **Version 1 (Principal/Avanzada)**: Ofrece una funcionalidad completa con múltiples fuentes de datos (Elasticsearch, DisGeNET, PDFs vía RAG con `HuggingFaceEmbeddings`, ClinVar API) y un agente con diversas herramientas.
*   **Version 2 (Variante/Simplificada)**: Se enfoca en RAG a partir de documentos locales (PDF, CSV, JSON) utilizando `OllamaEmbeddings` y un agente más simple.

Para una explicación detallada de las arquitecturas y diferencias, consulta el archivo `API_VERSIONS_EXPLAINED.md`.

<!--
    La imagen de arquitectura actual podría necesitar una revisión o ser reemplazada para reflejar ambas versiones.
    ![Arquitectura del Chatbot](/documentos_y_matcomplement/diagramas_graficos/readme.jpg)
-->

## Requisitos Previos
- Python 3.8 o superior (actualizado para compatibilidad con dependencias recientes de Langchain)
- Redis (opcional, para caché en las aplicaciones Streamlit)
- Ollama (para la Version 2, si se usan `OllamaEmbeddings` localmente)
- Elasticsearch (para la Version 1)

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/nombre-del-repositorio.git # Reemplaza con tu URL de repo
    cd nombre-del-repositorio
    ```

2.  **Crear un entorno virtual e instalar dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
    A continuación, instala las dependencias para la versión que desees utilizar (o ambas):

    **Para Version 1 (Principal/Avanzada):**
    ```bash
    pip install -r requirements/requirements_v1_main.txt
    ```
    **Para Version 2 (Variante/Simplificada):**
    ```bash
    pip install -r requirements/requirements_v2_variant.txt
    ```

## Configuración de Variables de Entorno

Crea un archivo `.env` en el directorio raíz del proyecto y añade las siguientes variables según sea necesario:

### Comunes para Ambas Versiones
*   `GROQ_API_KEY`: Tu clave API de Groq (requerida).
*   `HUGGING_FACE_API_TOKEN`: Tu token de Hugging Face API (usado por V1 para `HuggingFaceEmbeddings`).
*   `STREAMLIT_REDIS_URL`: URL de tu instancia Redis para la caché de las apps Streamlit (ej. `redis://localhost:6379`). Ambas apps Streamlit pueden usar esta variable y gestionarán namespaces de caché distintos (`main_cache` y `streamlit_variant_cache`).

### Específicas para Version 1 (Principal/Avanzada - `api/main.py`)
*   `GROQ_GENERAL_MODEL_NAME`: (Opcional) Modelo Groq para tareas generales (defecto: `mixtral-8x7b-32768`).
*   `GROQ_MEDICAL_MODEL_NAME`: (Opcional) Modelo Groq para tareas médicas (defecto: `mixtral-8x7b-32768`).
*   `PDF_DIRECTORY_PATH`: Ruta al directorio con PDFs para el RAG de V1 (ej. `./data/main_app_pdfs/`).
*   `ELASTICSEARCH_ENDPOINT`: Endpoint de Elasticsearch (ej. `tu-endpoint.es.cloud.region.amazonaws.com`).
*   `ELASTICSEARCH_USERNAME`: Usuario de Elasticsearch.
*   `ELASTICSEARCH_PASSWORD`: Contraseña de Elasticsearch.
*   `MAIN_API_URL`: (Opcional, para Streamlit V1) URL del backend V1 (defecto: `http://localhost:8000/ask`).

### Específicas para Version 2 (Variante/Simplificada - `api/main_variant_ollama.py`)
*   `OLLAMA_BASE_URL`: (Opcional) URL base si tu instancia Ollama no corre en `http://localhost:11434`.
*   `VARIANT_PDF_PATH`: (Opcional) Ruta al directorio de PDFs para el RAG de V2 (defecto: `./data/variant_docs/pdf/`).
*   `VARIANT_CSV_PATH`: (Opcional) Ruta al archivo CSV para el RAG de V2 (defecto: `./data/variant_docs/docs.csv`).
*   `VARIANT_JSON_PATH`: (Opcional) Ruta al archivo JSON para el RAG de V2 (defecto: `./data/variant_docs/docs.json`).
*   `VARIANT_API_URL`: (Opcional, para Streamlit V2) URL del backend V2 (defecto: `http://localhost:8001/ask`).

## Caché

Ambas aplicaciones FastAPI (`main.py` y `main_variant_ollama.py`) utilizan `aiocache` para caché en memoria por defecto para ciertas operaciones.
Las aplicaciones Streamlit (`streamlit_app.py` y `streamlit_variant_app.py`) pueden usar Redis para una caché más persistente y compartida si `STREAMLIT_REDIS_URL` está configurado. Cada app Streamlit usa un namespace de caché diferente para evitar colisiones.

## Uso y Ejecución

Asegúrate que los servicios externos necesarios (Elasticsearch para V1, Ollama para V2) estén en ejecución.

### Ejecutar Version 1 (Principal/Avanzada)

1.  **Iniciar Backend V1 (`api/main.py`):**
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
2.  **Iniciar Frontend V1 (`api/streamlit_app.py`):**
    (En una nueva terminal)
    ```bash
    streamlit run api/streamlit_app.py
    ```

### Ejecutar Version 2 (Variante/Simplificada)

1.  **Iniciar Backend V2 (`api/main_variant_ollama.py`):**
    ```bash
    uvicorn api.main_variant_ollama:app --host 0.0.0.0 --port 8001 --reload
    ```
2.  **Iniciar Frontend V2 (`api/streamlit_variant_app.py`):**
    (En una nueva terminal)
    ```bash
    streamlit run api/streamlit_variant_app.py
    ```

## Configuración de Datos

*   **Version 1 (`main.py`):**
    *   Coloca los documentos PDF en el directorio especificado por `PDF_DIRECTORY_PATH`.
    *   Asegúrate que Elasticsearch esté poblado con datos en el índice `genetic_information` si la funcionalidad "avanzada" que lo requiere será utilizada.
*   **Version 2 (`main_variant_ollama.py`):**
    *   Coloca los documentos PDF en el directorio `VARIANT_PDF_PATH`.
    *   Coloca el archivo CSV en `VARIANT_CSV_PATH`.
    *   Coloca el archivo JSON en `VARIANT_JSON_PATH`.

## Arquitectura y Flujo de Datos

A continuación, se presentan diagramas simplificados para cada versión. Para una explicación más exhaustiva, consulta `API_VERSIONS_EXPLAINED.md`.

### Version 1 (Principal/Avanzada) - Flujo Simplificado

```mermaid
graph TD
    A[Usuario acceda a Streamlit V1 (streamlit_app.py)] --> B{Ingresa Pregunta}
    B --> C[API V1: main.py (/ask)]
    C --> D{Clasifica Pregunta}

    D -- "Básico" --> E[Agente Langchain (custom_agent.py)]
    E --> F[Herramientas: Wikipedia, Arxiv, Mayo Clinic, ClinVar API]
    F --> G[LLM ChatGroq (General)]
    G --> H[Respuesta Procesada]

    D -- "Intermedio" --> I[RAG desde PDFs (FAISS)]
    I --> J[Contexto de Documentos]
    J --> K[LLM ChatGroq (Médico)]
    K --> H

    D -- "Avanzado" --> L[Consulta Elasticsearch y ClinVar API (directo)]
    L --> M[Contexto Estructurado]
    M --> K

    H --> N[Traduce al Español (si es necesario)]
    N --> O[Respuesta a Streamlit V1]
    O --> P[Muestra Respuesta al Usuario]
```

### Version 2 (Variante/Simplificada) - Flujo Simplificado

```mermaid
graph TD
    A_v2[Usuario acceda a Streamlit V2 (streamlit_variant_app.py)] --> B_v2{Ingresa Pregunta}
    B_v2 --> C_v2[API V2: main_variant_ollama.py (/ask)]

    C_v2 --> D_v2[Vector Store FAISS (PDF, CSV, JSON) con OllamaEmbeddings]
    D_v2 --> E_v2[Herramienta Retriever de Documentos]

    C_v2 --> F_v2[Agente Langchain (custom_agent_groq_variant.py - create_custom_tools_agent)]

    F_v2 --> G_v2{Herramientas del Agente}
    G_v2 --> E_v2
    G_v2 --> H_v2[Wikipedia]
    G_v2 --> I_v2[Mayo Clinic (Scraper Local)]
    G_v2 --> J_v2[ClinVar (Scraper Local)]

    F_v2 --> K_v2[LLM ChatGroq (ydshieh/tiny-random-gptj-for-question-answering)]
    K_v2 --> L_v2[Respuesta Procesada]
    L_v2 --> M_v2[Traduce al Español (si es necesario)]
    M_v2 --> N_v2[Respuesta a Streamlit V2]
    N_v2 --> O_v2[Muestra Respuesta al Usuario]
```

Para un entendimiento detallado del flujo y la arquitectura de cada versión, por favor consulta `API_VERSIONS_EXPLAINED.md`.

## Contacto y Soporte
Si tienes alguna pregunta o encuentras algún problema, no dudes en abrir un *issue* en este repositorio.

¡Disfruta usando los chatbots!
