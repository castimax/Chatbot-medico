import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DISGENET_API_KEY = os.getenv("DISGENET_API_KEY")

if not DISGENET_API_KEY:
    raise ValueError("DISGENET_API_KEY not found in environment variables.")

API_HOST = "https://www.disgenet.org/api"

# Create a single session for DisGeNET API requests
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {DISGENET_API_KEY}"})

def get_disease_associated_genes(disease_id: str, source: str = 'UNIPROT'):
    """
    Retrieves genes associated with a specific disease CUI.
    Maps to DisGeNET endpoint: /gda/disease/{disease_id}
    (This was formerly get_genes_associated_with_disease)
    """
    try:
        response = session.get(f"{API_HOST}/gda/disease/{disease_id}", params={'source': source})
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from DisGeNET for disease {disease_id}: {e}")
        return None # Or raise a custom exception

def get_gene_associated_diseases(gene_id: str, source: str = 'UNIPROT'):
    """
    Retrieves diseases associated with a specific gene.
    Placeholder: Actual DisGeNET endpoint for gene-to-disease might differ,
    e.g., /gda/gene/{gene_id} or similar.
    This requires checking DisGeNET API documentation.
    For now, this will return a placeholder or error.
    """
    # Example: Assuming the endpoint is /gda/gene/{gene_id}
    # This is a guess and needs verification with DisGeNET documentation.
    try:
        # Let's assume the gene_id could be an NCBI gene ID or symbol.
        # The DisGeNET API might require a specific type of gene identifier.
        response = session.get(f"{API_HOST}/gda/gene/{gene_id}", params={'source': source})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from DisGeNET for gene {gene_id}: {e}")
        # Depending on requirements, could return empty list, None, or re-raise
        return {"error": f"Could not fetch diseases for gene {gene_id}. Endpoint functionality needs verification.", "details": str(e)}

if __name__ == '__main__':
    # Example Usage (Optional: can be kept for direct script execution for testing)
    print("Starting DisGeNET API tests...")

    if not DISGENET_API_KEY or DISGENET_API_KEY == "YOUR_API_KEY_HERE":
        print("Skipping live API tests as DISGENET_API_KEY is not configured or is a placeholder.")
    else:
        # Test for get_disease_associated_genes
        test_disease_id = "C0002395"  # Alzheimer's Disease
        print(f"\nTesting get_disease_associated_genes for disease: {test_disease_id}")
        genes_data = get_disease_associated_genes(test_disease_id)
        if genes_data:
            print(f"Found {len(genes_data)} associations for {test_disease_id}.")
            # print(json.dumps(genes_data[:2], indent=2)) # Print first 2 results
        else:
            print(f"No gene data found or error for disease {test_disease_id}.")

        # Test for get_gene_associated_diseases
        # Using a common gene symbol, e.g., APOE (associated with Alzheimer's)
        # DisGeNET might use NCBI Gene IDs (e.g., 348 for APOE) or UniProt IDs.
        # For this example, let's try with a known UniProt ID if the API supports it, or gene symbol.
        # The API documentation should clarify what identifier types are supported.
        test_gene_id = "APOE" # Common gene symbol, might need to be an ID like "348"
        # Or, using a UniProt ID if that's what the (assumed) endpoint takes:
        # test_gene_id = "P02649" # APOE UniProt ID

        print(f"\nTesting get_gene_associated_diseases for gene: {test_gene_id}")
        diseases_data = get_gene_associated_diseases(test_gene_id) # Using default source 'UNIPROT'
        if diseases_data and not diseases_data.get("error"):
            print(f"Found {len(diseases_data)} associations for gene {test_gene_id}.")
            # print(json.dumps(diseases_data[:2], indent=2)) # Print first 2 results
        else:
            print(f"No disease data found or error for gene {test_gene_id}: {diseases_data}")

    print("\nDisGeNET API tests finished.")
