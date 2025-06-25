"""
FastAPI application for a variant RAG (Retrieval Augmented Generation) pipeline.

Features:
- Uses ChatGroq for the language model.
- Uses OllamaEmbeddings for document embeddings.
- Loads documents from PDF, CSV, and JSON sources into a FAISS vector store.
- Provides an /ask endpoint that uses a Langchain agent with tools:
    - Retriever tool for RAG from loaded documents (PDF, CSV, JSON).
    - Wikipedia.
    - Custom Mayo Clinic web scraper (local implementation).
    - Custom ClinVar web scraper (local implementation).
- Intended to be a simpler, alternative RAG setup compared to the main application.
Environment Variables:
- GROQ_API_KEY: Required for ChatGroq.
- OLLAMA_BASE_URL (implicitly by OllamaEmbeddings): If Ollama runs elsewhere.
- PDF_DIRECTORY_PATH, CSV_FILE_PATH, JSON_FILE_PATH (conceptual): Paths for document loading need to be configured.
"""
import os
from fastapi import FastAPI, Request
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader, JSONLoader
from dotenv import load_dotenv
from googletrans import Translator
from langchain_community.tools import WikipediaQueryRun, Tool
from langchain.tools.retriever import create_retriever_tool # Import for RAG tool
from langchain_community.utilities import WikipediaAPIWrapper
# Use create_custom_tools_agent from the variant agent file
from custom_agent_groq_variant import create_custom_tools_agent
import requests
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

model_name = "ydshieh/tiny-random-gptj-for-question-answering"
model = ChatGroq(model_name=model_name, groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context provided.
    Provide the most accurate answer based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

def create_document_processing_pipeline():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OllamaEmbeddings()

    def load_and_split_documents():
        # TODO: Update these paths to be configurable (e.g., via environment variables) or use actual paths.
        # These are placeholder paths.
        pdf_document_path = os.getenv("VARIANT_PDF_PATH", "./data/variant_docs/pdf/")
        csv_document_path = os.getenv("VARIANT_CSV_PATH", "./data/variant_docs/docs.csv")
        json_document_path = os.getenv("VARIANT_JSON_PATH", "./data/variant_docs/docs.json")

        pdf_loader = PyPDFDirectoryLoader(pdf_document_path)
        pdf_docs = pdf_loader.load()

        # Ensure CSVLoader and JSONLoader point to actual files if used, or handle FileNotFoundError.
        csv_loader = CSVLoader(csv_document_path)
        csv_docs = csv_loader.load()

        json_loader = JSONLoader("./ruta/al/archivo.json")
        json_docs = json_loader.load()

        docs = pdf_docs + csv_docs + json_docs
        return text_splitter.split_documents(docs)

    docs = load_and_split_documents()
    vectors = FAISS.from_documents(docs, embeddings)

    return vectors

@app.on_event("startup")
async def startup_event():
    app.state.vectors = create_document_processing_pipeline()

    wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    app.state.wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data["question"]

    translator = Translator()
    question_en = translator.translate(question, dest='en').text

    # Retrieve the vector store
    vectors = app.state.vectors
    if not vectors:
        # Fallback or error if vectors are not loaded
        return {"respuesta": "Error: El almacén de vectores de documentos no está disponible.", "contexto": []}

    # Create a retriever
    retriever = vectors.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    # Create a retriever tool
    document_retriever_tool = create_retriever_tool(
        retriever,
        name="document_knowledge_retriever",
        description="Busca y devuelve información relevante de los documentos cargados (PDF, CSV, JSON) para responder a la pregunta del usuario. Úsalo si la pregunta podría referirse a contenido específico de estos documentos."
    )

    # Define other tools for this variant
    mayo_clinic_tool = Tool(
        name="Mayo Clinic Web Scraper",
        func=lambda q: fetch_mayo_clinic_data_variant(q.replace(" ", "-").lower()),
        description="Busca información sobre enfermedades en el sitio web de Mayo Clinic (versión variante con scraping). Útil para síntomas generales, tratamientos de enfermedades comunes."
    )

    clinvar_tool_variant = Tool(
        name="ClinVar Web Scraper",
        func=fetch_clinvar_data_variant,
        description="Busca información genética en el sitio web de ClinVar (versión variante con scraping). Útil para preguntas sobre genes específicos o variantes genéticas."
    )

    tools = [app.state.wikipedia_tool, mayo_clinic_tool, clinvar_tool_variant, document_retriever_tool]

    # Consider adjusting the main prompt to encourage use of the document retriever if appropriate.
    # For now, relying on agent's ability to pick tools based on descriptions.
    agent = create_custom_tools_agent(model, tools, prompt)

    response = agent.invoke({'input': question_en})

    response_es = translator.translate(response['output'], dest='es').text

    return {"respuesta": response_es, "contexto": response.get("intermediate_steps", [])}

def fetch_mayo_clinic_data_variant(disease: str) -> str:
    """
    Fetches disease information from Mayo Clinic by scraping.
    This is a variant-specific implementation.
    """
    url = f"https://www.mayoclinic.org/diseases-conditions/{disease}/symptoms-causes/syc-20376178"
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
        soup = BeautifulSoup(page.content, 'html.parser')
        content_div = soup.find('div', {'class': 'content'})
        if content_div:
            return content_div.get_text(separator="\n").strip()
        return f"No se encontró contenido principal para {disease} en Mayo Clinic."
    except requests.exceptions.RequestException as e:
        return f"Error al contactar Mayo Clinic para {disease}: {e}"
    except Exception as e:
        return f"Error al procesar datos de Mayo Clinic para {disease}: {e}"


def fetch_clinvar_data_variant(gene: str) -> str:
    """
    Fetches gene information from ClinVar by scraping.
    This is a variant-specific implementation.
    """
    url = f"https://www.ncbi.nlm.nih.gov/clinvar/?term={gene}[gene]"
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        content_div = soup.find('div', {'class': 'docsum-content'}) # This class might be too generic or change
        if content_div:
            # Extracting more specific information might be needed.
            # For now, taking the whole text content of the first 'docsum-content'.
            return content_div.get_text(separator="\n").strip()
        return f"No se encontró contenido principal para el gen {gene} en ClinVar."
    except requests.exceptions.RequestException as e:
        return f"Error al contactar ClinVar para {gene}: {e}"
    except Exception as e:
        return f"Error al procesar datos de ClinVar para {gene}: {e}"
