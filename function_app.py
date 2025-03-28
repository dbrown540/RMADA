import os
import json
import re
import tempfile
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import azure.functions as func
import logging
import PyPDF2
from typing import List, Dict, Any, Union

# Azure Function specific configuration
app = func.FunctionApp()

# Download NLTK data if not already downloaded
nltk.download('punkt', quiet=True)

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def backup_embeddings(embeddings_file_path: str, connection_string: str, container_name: str, blob_name: str):
    """
    Upload the embeddings file to Azure Blob Storage for persistent backup.
    
    :param embeddings_file_path: Local path to the embeddings JSON file.
    :param connection_string: Azure Blob Storage connection string.
    :param container_name: Name of the Blob container.
    :param blob_name: Name to use for the blob in the container.
    """
    try:
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get or create the container
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception:
            # Container might already exist
            pass
        
        # Create a BlobClient and upload the file
        blob_client = container_client.get_blob_client(blob_name)
        with open(embeddings_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logging.info(f"Backup of embeddings file '{embeddings_file_path}' to container '{container_name}' as blob '{blob_name}' succeeded.")
    except Exception as e:
        logging.error(f"Error during backup of embeddings: {e}")

def save_embeddings(embeddings_path: str, 
                    document_title: str, 
                    sentences: List[str], 
                    document_embeddings: np.ndarray):
    """
    Save or update embeddings for a specific document.
    
    :param embeddings_path: Path to the embeddings JSON file
    :param document_title: Title of the document
    :param sentences: List of sentences
    :param document_embeddings: Numpy array of embeddings
    """
    try:
        # Load existing embeddings or create new dict
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
        else:
            embeddings_data = {}
        
        # Convert numpy array to list for JSON serialization
        embeddings_list = document_embeddings.tolist()
        
        # Store document embeddings
        embeddings_data[document_title] = {
            "sentences": sentences,
            "embeddings": embeddings_list
        }
        
        # Save updated embeddings
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logging.info(f"Saved embeddings for {document_title}: {len(sentences)} sentences")
    
    except Exception as e:
        logging.error(f"Error saving embeddings for {document_title}: {e}")

def load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """
    Load existing embeddings from a JSON file.
    
    :param embeddings_path: Path to the embeddings JSON file
    :return: Dictionary of embeddings
    """
    try:
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return {}

class RMADAAnalyzer:
    def __init__(self, documents_dir: str, excel_path: str, mode: str = "test"):
        """
        Initialize the RMADA analyzer with configuration parameters.
        
        :param documents_dir: Directory containing PDF documents
        :param excel_path: Path to the Excel file with requirements
        :param mode: Processing mode ('test' or 'prod')
        """
        self.DOCUMENTS_DIR = documents_dir
        self.EXCEL_PATH = excel_path
        self.MODE = mode
        
        # Use current working directory for output files
        self.OUTPUT_JSON = "all_sentences.json"
        self.RESULTS_JSON = "requirement_matches.json"
        self.EMBEDDINGS_JSON = "document_embeddings.json"
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Sentence transformer model (lazy loaded)
        self._model = None
        
    def extract_sentences(self, file_path: str) -> List[str]:
        """Extract all sentences from a document file."""
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    sentences = sent_tokenize(content)
                    return sentences
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return []

    def _extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text content from a PDF file and split into sentences."""
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    text_content += page_text + " "
            
            if not text_content.strip():
                logging.warning(f"No text content extracted from {pdf_path}. The PDF might be scanned or image-based.")
                return []
            
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            try:
                sentences = sent_tokenize(text_content)
            except Exception as e:
                logging.warning(f"NLTK sentence tokenization failed: {e}. Falling back to basic sentence splitting.")
                sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', text_content) if s.strip()]
            
            sentences = [s for s in sentences if len(s.strip()) > 10]
            
            return sentences
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return []

    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the directory and create/load a master JSON file.
        """
        # If embedding file exists, load from there first
        if os.path.exists(self.EMBEDDINGS_JSON):
            try:
                with open(self.EMBEDDINGS_JSON, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                    # If embeddings exist, reconstruct the sentences JSON
                    all_docs = {
                        doc_title: {
                            "title": doc_title, 
                            "sentences": doc_data.get("sentences", [])
                        } 
                        for doc_title, doc_data in embeddings_data.items()
                    }
                    logging.info(f"Loaded existing sentences from {self.EMBEDDINGS_JSON}")
                    return all_docs
            except Exception as e:
                logging.warning(f"Error loading embeddings: {e}")
        
        # If no existing embeddings, process documents
        all_docs = {}
        
        if not os.path.exists(self.DOCUMENTS_DIR):
            logging.error(f"Directory {self.DOCUMENTS_DIR} not found!")
            return all_docs
        
        # Lazy load the model
        if self._model is None:
            logging.info("Loading sentence transformer model...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Model loaded!")
        
        # Find all PDF files
        pdf_files = [
            os.path.join(root, file) 
            for root, _, files in os.walk(self.DOCUMENTS_DIR) 
            for file in files if file.lower().endswith('.pdf')
        ]
        
        logging.info(f"Found {len(pdf_files)} PDF documents to process")
        
        # Process each PDF file
        for idx, file_path in enumerate(pdf_files, 1):
            file_name = os.path.basename(file_path)
            logging.info(f"Processing document {idx}/{len(pdf_files)}: {file_name}")
            
            # Extract sentences from PDF
            sentences = self.extract_sentences(file_path)
            logging.info(f"Extracted {len(sentences)} sentences from {file_name}")
            
            # Store sentences
            all_docs[file_name] = {
                "title": file_name,
                "sentences": sentences
            }
            
            # Compute and save embeddings for this document
            document_embeddings = self._model.encode(sentences)
            save_embeddings(self.EMBEDDINGS_JSON, file_name, sentences, document_embeddings)

            # Backup the embeddings file
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            backup_embeddings(
                embeddings_file_path=self.EMBEDDINGS_JSON,
                connection_string="pastperformancev2rg8ee5",
                container_name="rmada-backups",
                blob_name="document_embeddings_backup.json"
            )
        
        # Save to sentences JSON
        with open(self.OUTPUT_JSON, 'w', encoding='utf-8') as json_file:
            json.dump(all_docs, json_file, indent=2)
        
        logging.info(f"Saved sentences to {self.OUTPUT_JSON}")
        return all_docs

    def load_requirements(self) -> pd.DataFrame:
        """Load requirements from the Excel file, assuming requirements are in column A (index 0)."""
        try:
            df = pd.read_excel(self.EXCEL_PATH)
            
            # Get the first column (column A or index 0) as requirements
            if len(df.columns) == 0:
                logging.error("Excel file has no columns!")
                return None
                
            # Get the first column name
            first_column = df.columns[0]
            logging.info(f"Using column '{first_column}' for requirements")
            
            # Check if the column exists and has data
            if df[first_column].isnull().all():
                logging.warning(f"Column '{first_column}' has no data")
            
            logging.info(f"Loaded {len(df)} requirements from Excel file")
            return df
            
        except Exception as e:
            logging.error(f"Error loading Excel file: {e}")
            return None

    def find_related_sentences(self, requirement: str, all_sentences: Dict[str, Any], all_titles: List[str]) -> List[Dict[str, Any]]:
        """Find the top three most related sentences using pre-computed embeddings."""
        # Lazy load the model if not already loaded
        if self._model is None:
            logging.info("Loading sentence transformer model...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Model loaded!")
        
        # Load pre-computed embeddings
        embeddings_data = load_embeddings(self.EMBEDDINGS_JSON)
        
        # Encode the requirement
        req_embedding = self._model.encode([requirement])[0]
        
        # Prepare lists for all sentences and their embeddings
        all_sentence_list = []
        doc_titles = []
        all_sentence_embeddings = []
        
        # Iterate through loaded embeddings
        for doc_title, doc_data in embeddings_data.items():
            sentences = doc_data.get('sentences', [])
            embeddings = doc_data.get('embeddings', [])
            
            all_sentence_list.extend(sentences)
            doc_titles.extend([doc_title] * len(sentences))
            all_sentence_embeddings.extend(embeddings)
        
        # Convert to numpy array for cosine similarity
        sentence_embeddings_array = np.array(all_sentence_embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [req_embedding], 
            sentence_embeddings_array
        )[0]
        
        # Get indices of top 3 most similar sentences
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        # Prepare top matches
        top_matches = [
            {
                "sentence": all_sentence_list[idx],
                "document": doc_titles[idx],
                "similarity_score": float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return top_matches

    def assess_relevance(self, requirement: str, matches: List[Dict[str, Any]]) -> int:
        """
        Use OpenAI to assess the relevance of matches to the requirement.
        
        :param requirement: The requirement to assess
        :param matches: List of matching sentences with metadata
        :return: Integer score (0, 1, or 2) representing relevance
        """
        # Prepare matches text
        matches_text = ""
        for i, match in enumerate(matches, 1):
            matches_text += f"Match {i} (from document '{match['document']}'):\n{match['sentence']}\n\n"
        
        # Prepare the prompt
        prompt = f"""
        Assess the relevance of the following past performance examples to the given requirement:
        
        Requirement: {requirement}
        
        Past Performance Examples:
        {matches_text}
        
        Rate the relevance on the following scale:
        0 - No relevant experience
        1 - Some/Indirect Experience (similar work but not for CMS/CMMI, or work for CMS/CMMI but tangential to the RMADA requirements)
        2 - Significant Relevant Experience
                
        Provide ONLY a single integer (0, 1, or 2) with no explanation or additional text.
        """
        
        # Add retry mechanism for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Query OpenAI 
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in evaluating past performance relevance to requirements for government RFIs. You must respond with ONLY a single number (0, 1, or 2) with no explanation."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract response content
                response_text = response.choices[0].message.content.strip()
                
                # Check if the response is a valid integer (0, 1, or 2)
                if response_text in ["0", "1", "2"]:
                    return int(response_text)
                else:
                    logging.warning(f"Invalid response format: '{response_text}'. Expected '0', '1', or '2'. Retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logging.error(f"Failed to get valid response after {max_retries} attempts. Last response: '{response_text}'")
                        return 0  # Default to 0 if all retries fail
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"OpenAI API error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(f"Failed to get response from OpenAI after {max_retries} attempts: {e}")
                    return 0  # Default to 0 if all retries fail
    
    def process_requirements(self, requirements_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process requirements and find matches.
        
        :param requirements_df: DataFrame containing requirements
        :return: List of dictionaries with requirement matches and assessments
        """
        all_docs = self.process_all_documents()
        
        if not all_docs:
            logging.error("No documents processed.")
            return []
        
        results = []
        
        # Limit processing if in test mode
        if self.MODE == "test":
            requirements_to_process = requirements_df.head(10)
        else:
            requirements_to_process = requirements_df
        
        logging.info(f"Processing {len(requirements_to_process)} requirements")
        
        # Get the first column name
        first_column = requirements_to_process.columns[0]
        
        for idx, row in requirements_to_process.iterrows():
            requirement = row[first_column]
            logging.info(f"Processing requirement {idx+1}/{len(requirements_to_process)}: {requirement[:50]}...")
            
            # Find related sentences
            matches = self.find_related_sentences(requirement, all_docs, list(all_docs.keys()))
            
            # Assess relevance using OpenAI - now returns an integer
            relevance_score = self.assess_relevance(requirement, matches)
            
            # Create a human-readable assessment message based on the score
            if relevance_score == 0:
                assessment = "0 - No relevant experience"
            elif relevance_score == 1:
                assessment = "1 - Some/Indirect Experience (similar work but not for CMS/CMMI, or work for CMS/CMMI but tangential to the RMADA requirements)"
            elif relevance_score == 2:
                assessment = "2 - Significant Relevant Experience"
            else:
                assessment = f"Invalid score: {relevance_score}"
            
            # Store results
            result = {
                "requirement": requirement,
                "matches": matches,
                "relevance_score": relevance_score,  # Store the numeric score
                "assessment": assessment,  # Store the human-readable assessment
                "excel_row_index": idx
            }
            results.append(result)
        
        return results

@app.route(route="rmada_trigger")
def rmada_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('RMADA document analysis function triggered.')
    
    try:
        excel_file = req.params.get("excelFile", None)
        # Hardcode the documents directory and excel file path
        documents_dir = os.path.join(os.getcwd(), "Documents", "sows")
        excel_path = os.path.join(os.getcwd(), "Documents", "matrices", excel_file)
        
        # Only mode is specified in the request (default to 'test' if not provided)
        mode = req.params.get('mode', 'test')
        try:
            req_body = req.get_json()
            mode = req_body.get('mode', mode)
        except ValueError:
            pass
        
        logging.info(f"Running in {mode} mode")
        
        # Initialize analyzer
        analyzer = RMADAAnalyzer(documents_dir, excel_path, mode)
        
        # Load requirements
        requirements_df = analyzer.load_requirements()
        if requirements_df is None or len(requirements_df) == 0:
            return func.HttpResponse(
                "Failed to load requirements or no requirements found.",
                status_code=400
            )
        
        logging.info(f"Successfully loaded requirements: {len(requirements_df)} rows")
        
        # Process requirements - verify the method exists
        if not hasattr(analyzer, 'process_requirements'):
            logging.error("Method 'process_requirements' not found in RMADAAnalyzer class!")
            return func.HttpResponse(
                "Internal server error: Method 'process_requirements' not found.",
                status_code=500
            )
        
        # Call the method
        results = analyzer.process_requirements(requirements_df)
        
        # Save results to file
        output_file = f"test_{analyzer.RESULTS_JSON}" if mode == "test" else analyzer.RESULTS_JSON
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=2)
        
        logging.info(f"Results processed and saved to {output_file}")
        
        # Read the content of the results file and return it as the response
        with open(output_file, 'r', encoding='utf-8') as json_file:
            results_content = json_file.read()
        
        return func.HttpResponse(
            results_content,
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"Error in RMADA trigger: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main entry point for the Azure Function.
    Delegates to the rmada_trigger function.
    """
    return rmada_trigger(req)

# Add any additional setup or initialization if needed
if __name__ == "__main__":
    # This block is typically used for local testing or script initialization
    # You might want to add any necessary setup here
    pass