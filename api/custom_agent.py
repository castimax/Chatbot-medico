"""
Core agent logic for the main application.

This module defines the tools and agent components used by the main FastAPI
application (`main.py`). It includes:
- Tools for Wikipedia, Arxiv, Mayo Clinic (web scraping), and ClinVar (API).
- Functions to create and run a Langchain agent using these tools with a ChatGroq LLM.
- Utility function to search data in Elasticsearch.

The Mayo Clinic tool caches scraped data in a local JSON file (`./data/clinica_mayo.json`).
"""
import os
import json
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader # Keep for main.py context if needed elsewhere, though not directly used in this file
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep for main.py context if needed elsewhere
from elasticsearch import Elasticsearch, helpers # Keep for search_data_in_es
from langchain_groq import ChatGroq # Add for new model
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
import datetime # Added import for timestamping

# Definición de la ruta del archivo JSON (for Mayo Clinic data caching)
# This might be shared with the variant or be distinct. Assuming shared for now.
json_path = './data/clinica_mayo.json'

# Función para guardar datos en la ruta definida
def save_clinica_mayo_data(data):
    # Consider adding error handling for file operations
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4) # Added indent for readability

# Función para cargar datos desde la ruta definida
def load_clinica_mayo_data():
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty list if JSON is corrupted
    return []

# Función para realizar web scraping en Mayo Clinic (Main Version)
def fetch_clinica_mayo_data(disease: str) -> str:
    url = f"https://www.mayoclinic.org/diseases-conditions/{disease}/symptoms-causes/syc-20376178" # Updated URL
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        description_tag = soup.find('div', {'class': 'content'}) # Assuming 'content' class is primary target
        if description_tag:
            description = description_tag.get_text().strip()
            # Caching logic (optional, can be part of the tool's responsibility or app level)
            # For now, keeping the local JSON cache as per original structure
            data = load_clinica_mayo_data()
            # Avoid duplicate entries if necessary, e.g., by checking if disease entry already exists
            data.append({
                "disease": disease,
                "source": "Mayo Clinic",
                "description": description,
                "timestamp": str(datetime.datetime.now()) # Optional: add timestamp
            })
            save_clinica_mayo_data(data)
            return description
        return f"No se encontró contenido principal para {disease} en Mayo Clinic."
    except requests.exceptions.RequestException as e:
        return f"Error al contactar Mayo Clinic para {disease}: {e}"
    except Exception as e:
        return f"Error al procesar datos de Mayo Clinic para {disease}: {e}"

# Copied from custom_agent_groq_variant.py
# Función para obtener datos de ClinVar usando su API
def fetch_clinvar_data(gene: str) -> str:
    api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term={gene}[gene]&retmode=json"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'esearchresult' in data and 'idlist' in data['esearchresult'] and data['esearchresult']['idlist']:
            ids = data['esearchresult']['idlist']
            summaries = []
            # Limit number of IDs to process to avoid very long requests/responses
            for clinvar_id in ids[:5]: # Process up to 5 results
                summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=clinvar&id={clinvar_id}&retmode=json"
                summary_response = requests.get(summary_url, timeout=10)
                summary_response.raise_for_status()
                summary_data = summary_response.json()
                if 'result' in summary_data and clinvar_id in summary_data['result']:
                    # Extract relevant info, e.g., title or other fields
                    summary_title = summary_data['result'][clinvar_id].get('title', 'No title')
                    # Potentially more details: clinical_significance, last_evaluated, etc.
                    summaries.append(summary_title)
            return "\n".join(summaries) if summaries else f"No se encontraron resúmenes para los IDs de ClinVar para {gene}."
        return f"No se encontraron IDs de ClinVar para el gen {gene}."
    except requests.exceptions.RequestException as e:
        return f"Error al contactar API de ClinVar para {gene}: {e}"
    except json.JSONDecodeError:
        return f"Error al decodificar respuesta JSON de ClinVar para {gene}."
    except Exception as e:
        return f"Error al procesar datos de ClinVar para {gene}: {e}"

# Función para crear el agente con herramientas personalizadas
def create_custom_tools_agent(model, tools, prompt):
    # datetime import is at the top of the file
    return initialize_agent(
        tools=tools,
        llm=model,
        agent_type="chat-conversational-react-description",
        prompt_template=prompt
    )

# Definir las herramientas personalizadas
def get_custom_tools():
    user_agent = 'MyApp/1.0 (example@example.com)'
    wikipedia_api_wrapper = WikipediaAPIWrapper(
        language='es', top_k_results=1, doc_content_chars_max=200, user_agent=user_agent)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

    arxiv_api_wrapper = ArxivAPIWrapper(
        top_k_results=1, doc_content_chars_max=200)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

    clinica_mayo_tool = Tool(
        name="Mayo Clinic",
        func=lambda q: fetch_clinica_mayo_data(q.replace(" ", "-").lower()),
        description="Busca información sobre enfermedades en el sitio web de Mayo Clinic."
    )

    clinvar_tool = Tool(
        name="ClinVar API", # Distinguish from variant's web scraper
        func=fetch_clinvar_data, # Uses the API version of fetch_clinvar_data
        description="Busca información genética en la API de ClinVar."
    )

    return [wikipedia_tool, arxiv_tool, clinica_mayo_tool, clinvar_tool]

# Función para procesar la consulta del usuario y devolver la respuesta adecuada
# This function provides a self-contained way to use the agent with its default settings.
# main.py might call create_custom_tools_agent directly for more control.
def process_query(query, lang='en'):
    tools = get_custom_tools()
    translator = Translator()

    # Traducir la pregunta al inglés si es necesario
    query_en = translator.translate(query, dest='en').text if lang == 'es' else query

    # Standardized to ChatGroq, using a common model, e.g., mixtral
    # Ensure GROQ_API_KEY is set in the environment
    model = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768"),
        groq_api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.7
    )

    # General prompt template for this agent's direct use
    prompt_template_str = """
    Utiliza la siguiente información y herramientas para responder la pregunta del usuario.
    Si no sabes la respuesta, simplemente di que no lo sabes, no inventes una respuesta.

    Contexto de herramientas: {context}
    Pregunta: {input}

    Devuelve solo la respuesta útil a continuación y nada más.
    Respuesta útil:
    """
    # Note: The agent type "chat-conversational-react-description" might manage context differently.
    # The prompt passed to initialize_agent is often a system message.
    # For direct LLM calls, the prompt structure is as above.
    # For agent, the prompt is more about setting its behavior.
    # Using a simpler prompt for the agent setup.
    # The actual prompt used by the agent's LLM part will be formed dynamically by the agent.

    # The prompt for create_custom_tools_agent is a template that the agent will use.
    # It typically includes placeholders for {input}, {chat_history}, {agent_scratchpad}.
    # For "chat-conversational-react-description", a specific prompt structure is expected by default.
    # We can provide a custom one if needed, but let's use the default behavior for now by passing a simple instruction.
    # The ChatPromptTemplate.from_template used here is more for direct model calls.
    # Let's keep the original prompt structure for process_query's direct agent setup.
    prompt = ChatPromptTemplate.from_template(prompt_template_str)


    agent = create_custom_tools_agent(model, tools, prompt) # Pass the prompt template object

    # Ejecutar la consulta utilizando el agente
    # The 'context' here would be from chat history if this were a conversational agent.
    # For a react agent, context is built via tool use.
    response = agent.invoke({'input': query_en, 'context': ''}) # Provide empty context if not conversational here

    # Traducir la respuesta de vuelta al español si es necesario
    response_es = translator.translate(response['output'], dest='es').text if lang == 'es' else response['output']

    return response_es

# Función para buscar datos en Elasticsearch
def search_data_in_es(index_name, query):
    es_url = f"https://{os.getenv('ELASTICSEARCH_USERNAME')}:{os.getenv('ELASTICSEARCH_PASSWORD')}@{os.getenv('ELASTICSEARCH_ENDPOINT')}:443"
    es = Elasticsearch(es_url)
    search_query = {
        "query": {
            "match": {
                "disease": query
            }
        }
    }
    response = es.search(index=index_name, body=search_query)
    return response['hits']['hits']
