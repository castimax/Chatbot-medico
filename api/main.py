"""
Main FastAPI application for the Health and Wellness Assistant.

Features:
- Handles different types of questions ("basico", "intermedio", "avanzado").
- "basico": Uses a Langchain agent with tools (Wikipedia, Arxiv, Mayo Clinic, ClinVar API).
- "intermedio": Augments context with RAG from PDF documents (FAISS vector store)
                 before calling the LLM.
- "avanzado": Builds context from Elasticsearch and ClinVar API data before calling the LLM.
- Language Models: Uses ChatGroq (configurable model names via environment variables).
- Embeddings: Uses HuggingFaceEmbeddings for PDF processing.
- Vector Store: FAISS for PDF documents.
- Data Sources: PDFs, Elasticsearch, DisGeNET API, ClinVar API, Mayo Clinic (via custom_agent).
- Caching: In-memory cache for /ask endpoint responses.

Key Environment Variables:
- GROQ_API_KEY: Required for ChatGroq LLMs.
- GROQ_GENERAL_MODEL_NAME: Model for general queries (default: mixtral-8x7b-32768).
- GROQ_MEDICAL_MODEL_NAME: Model for medical queries (default: mixtral-8x7b-32768).
- PDF_DIRECTORY_PATH: Path to directory containing PDF files for FAISS store.
- HUGGING_FACE_API_TOKEN: For HuggingFaceEmbeddings.
- ELASTICSEARCH_ENDPOINT, ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD: For ES connection.
"""
import os
import time
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from aiocache import Cache
from aiocache.serializers import JsonSerializer
from googletrans import Translator
from langchain_groq import ChatGroq # Changed from HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings # Stays as HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# Import get_custom_tools for direct use, search_data_in_es, and optionally fetch_clinvar_data
from custom_agent import create_custom_tools_agent, get_custom_tools, search_data_in_es, fetch_clinvar_data as agent_fetch_clinvar_data
from disgenet import get_disease_associated_genes, get_gene_associated_diseases
from functools import lru_cache
import json
import requests
from elasticsearch import Elasticsearch, helpers

load_dotenv()

app = FastAPI()

# Configurar el caché correctamente
cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PDF_DIRECTORY_PATH = os.getenv("PDF_DIRECTORY_PATH")
HUGGING_FACE_API_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')
ELASTICSEARCH_ENDPOINT = os.getenv('ELASTICSEARCH_ENDPOINT')
ELASTICSEARCH_USERNAME = os.getenv('ELASTICSEARCH_USERNAME')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')

if not GROQ_API_KEY:
    raise ValueError(
        "No se ha configurado la clave API de GROQ en las variables de entorno.")
if not PDF_DIRECTORY_PATH:
    raise ValueError(
        "No se ha configurado el directorio de PDFs en las variables de entorno.")
if not HUGGING_FACE_API_TOKEN:
    raise ValueError(
        "No se ha configurado el token API de Hugging Face en las variables de entorno.")
if not ELASTICSEARCH_ENDPOINT:
    raise ValueError(
        "No se ha configurado el endpoint de Elasticsearch en las variables de entorno.")
if not ELASTICSEARCH_USERNAME:
    raise ValueError(
        "No se ha configurado el nombre de usuario de Elasticsearch en las variables de entorno.")
if not ELASTICSEARCH_PASSWORD:
    raise ValueError(
        "No se ha configurado la contraseña de Elasticsearch en las variables de entorno.")

# Configuración de Elasticsearch con credenciales
es_url = f"https://{ELASTICSEARCH_USERNAME}:{ELASTICSEARCH_PASSWORD}@{ELASTICSEARCH_ENDPOINT}:443"
es = Elasticsearch(es_url)

# Verificar la conexión a Elasticsearch
try:
    es.ping()
    print("Conexión exitosa a Elasticsearch")
except Exception as e:
    print(f"No se pudo establecer conexión: {e}")

# Crear índice en Elasticsearch
index_name = "genetic_information"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Función para cargar y dividir documentos PDF
def load_and_split_pdf(pdf_path):
    pdf_loader = PyPDFDirectoryLoader(pdf_path)
    docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    return final_documents

# Función para vectorizar documentos y almacenarlos
def vectorize_and_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(documents, embeddings)
    return vectors

@lru_cache
def load_documents_and_vectors():
    pdf_path = PDF_DIRECTORY_PATH
    documents = load_and_split_pdf(pdf_path)
    vectors = vectorize_and_store(documents)
    return vectors

@lru_cache
def get_general_model():
    return ChatGroq(
        model_name=os.getenv("GROQ_GENERAL_MODEL_NAME", "mixtral-8x7b-32768"),
        groq_api_key=GROQ_API_KEY,
        temperature=0.7
    )

@lru_cache
def get_medical_model():
    # Ideally, this would be a medically tuned model available via Groq.
    # Using a general model as a placeholder.
    return ChatGroq(
        model_name=os.getenv("GROQ_MEDICAL_MODEL_NAME", "mixtral-8x7b-32768"),
        groq_api_key=GROQ_API_KEY,
        temperature=0.7
    )

general_model = get_general_model()
medical_model = get_medical_model()

prompt_template = """
Utiliza la siguiente información para responder la pregunta del usuario.
Si no sabes la respuesta, simplemente di que no lo sabes, no inventes una respuesta.

Contexto: {context}
Pregunta: {input}

Devuelve solo la respuesta útil a continuación y nada más.
Respuesta útil:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def classify_question(question):
    basic_keywords = ["qué es", "síntomas", "tratamiento"]
    intermediate_keywords = ["prevención", "diagnóstico", "medicación"]
    advanced_keywords = ["gen", "mutación", "riesgo genético"]

    if any(keyword in question.lower() for keyword in basic_keywords):
        return "basico"
    elif any(keyword in question.lower() for keyword in intermediate_keywords):
        return "intermedio"
    elif any(keyword in question.lower() for keyword in advanced_keywords):
        return "avanzado"
    else:
        return "unknown"

def extract_disease_from_question(question):
    diseases = ["diabetes", "cáncer", "hipertensión", "fibrosis quística", "artritis", "asma", "alzheimer", "parkinson"]
    for disease in diseases:
        if disease.lower() in question.lower():
            return disease
    return "unknown"

# Removed local fetch_clinvar_data and process_clinvar_data.
# Will use agent_fetch_clinvar_data from custom_agent.py for the "avanzado" path if raw data is needed,
# or rely on the ClinVar tool within the agent for "basico" path.

@app.on_event("startup")
async def startup_event():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = load_documents_and_vectors()
    app.state.vectors = vectors

@app.post("/ask")
async def ask_question(request: Request):
    start_time = time.time()

    data = await request.json()
    question = data.get("question")

    if not question:
        return {"error": "No se proporcionó pregunta."}

    translator = Translator()
    question_en = translator.translate(question, dest='en').text

    # Clasificar la pregunta
    question_type = classify_question(question_en)

    if question_type == "avanzado":
        # Buscar información técnica en Elasticsearch y ClinVar
        # Assuming 'disease' extracted could be a disease name or a gene symbol for advanced queries.
        term_for_search = extract_disease_from_question(question_en) # This might be a disease or gene
        
        es_hits = search_data_in_es(index_name, term_for_search)
        es_context_list = [
            f"ES Result: Disease: {hit['_source'].get('disease', 'N/A')}, Gene: {hit['_source'].get('gene', 'N/A')}, Mutation: {hit['_source'].get('mutation', 'N/A')}, Description: {hit['_source'].get('description', 'N/A')}"
            for hit in es_hits
        ]
        context = "\n".join(es_context_list)
        
        # For ClinVar, agent_fetch_clinvar_data expects a gene symbol.
        # If term_for_search is a disease, we might need to find associated genes first.
        # For simplicity, if term_for_search is considered a gene for ClinVar:
        # This is a simplification; robust gene extraction/mapping from question needed.
        clinvar_text_summary = agent_fetch_clinvar_data(term_for_search) # term_for_search ideally is a gene here
        if clinvar_text_summary and "No se encontró información" not in clinvar_text_summary and "No se encontraron IDs" not in clinvar_text_summary :
            context += f"\nClinVar Summary for {term_for_search}:\n{clinvar_text_summary}"
        
        model = medical_model
    elif question_type == "intermedio":
        # Usar información de la Clínica Mayo y la guía médica en PDF (vector store)
        disease_extracted = extract_disease_from_question(question_en)
        # Example: Augment with FAISS context for intermediate questions
        sim_docs = app.state.vectors.similarity_search(f"Información sobre {disease_extracted}")
        pdf_context = "\n".join([doc.page_content for doc in sim_docs])
        context = f"Información sobre {disease_extracted} de fuentes confiables y documentos:\n{pdf_context}"
        model = medical_model
    elif question_type == "basico":
        # Usar el agente con herramientas (Wikipedia, Arxiv, Mayo Clinic, ClinVar API)
        # Cache the final response dictionary
        cached_response_data = await cache.get(question_en)
        if cached_response_data and all(k in cached_response_data for k in ['answer', 'context', 'processing_time']):
            return cached_response_data

        tools = get_custom_tools() # Get tools from custom_agent.py
        agent_model = general_model # Use the general_model (ChatGroq)
        
        # The prompt for the agent is defined in main.py for this path
        agent = create_custom_tools_agent(agent_model, tools, prompt)

        agent_response = agent.invoke({'input': question_en})

        response_es = translator.translate(agent_response['output'], dest='es').text
        intermediate_steps = agent_response.get("intermediate_steps", [])

        end_time = time.time()
        processing_time = end_time - start_time

        final_response_data = {"answer": response_es, "context": intermediate_steps, "processing_time": processing_time}
        await cache.set(question_en, final_response_data, ttl=3600) # Cache the whole dict

        return final_response_data
    else:
        # Si no se puede clasificar la pregunta, usar el modelo general para intentar responder
        model = general_model
        input_data = {
            "input": question_en,
            "context": "No se pudo clasificar la pregunta. Intenta proporcionar una respuesta general."
        }
        response = model.invoke(input_data)

        response_es = translator.translate(response['output'], dest='es').text

        if "no lo sé" in response_es.lower() or "no entiendo" in response_es.lower():
            response_es = "Lo siento, no entiendo la pregunta. Por favor, intente formularla de otra manera."

        end_time = time.time()
        processing_time = end_time - start_time

        return {"answer": response_es, "context": input_data["context"], "processing_time": processing_time}

    # Generar la respuesta usando el modelo seleccionado
    input_data = {
        "input": question_en,
        "context": context
    }
    response = model.invoke(input_data)

    response_es = translator.translate(response['output'], dest='es').text

    end_time = time.time()
    processing_time = end_time - start_time

    return {"answer": response_es, "context": context, "processing_time": processing_time}

@app.get("/disease_genes/{gene_id}")
async def disease_genes(gene_id: str):
    data = get_disease_associated_genes(gene_id)
    return data

@app.get("/gene_diseases/{disease_id}")
async def gene_diseases(disease_id: str):
    data = get_gene_associated_diseases(disease_id)
    return data

@cache.cached(ttl=3600)
async def load_documents():
    try:
        pdf_loader = PyPDFDirectoryLoader(PDF_DIRECTORY_PATH)
        return pdf_loader.load()
    except Exception as e:
        print(f"Error cargando documentos: {e}")
        return []
