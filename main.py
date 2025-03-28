import os
import json
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from tqdm import tqdm  # For progress bars
import PyPDF2  # For reading PDF files

# Download NLTK data if not already downloaded
nltk.download('punkt', quiet=True)

# Define paths and settings
DOCUMENTS_DIR = r"C:\Users\Doug Brown\Desktop\Dannys Stuff\Job\Capability Matrix Automation\Documents\rfis\pdf"
EXCEL_PATH = "RMADA3_SOFTDEV.xlsx"
OUTPUT_JSON = "all_sentences.json"
RESULTS_JSON = "requirement_matches.json"
CHECKPOINT_FILE = "checkpoint.json"  # Checkpoint file to save progress

# Runtime mode: "test" processes only first 10 requirements, "prod" processes all
MODE = "test"  # Change to "prod" for full processing

# Initialize OpenAI client
client = OpenAI()


def extract_sentences(file_path):
    """Extract all sentences from a document file."""
    try:
        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        else:
            # For text-based files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                # Use NLTK's sentence tokenizer
                sentences = sent_tokenize(content)
                return sentences
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file and split into sentences."""
    try:
        text_content = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""  # Handle None returns
                text_content += page_text + " "
        
        # Check if any text was extracted
        if not text_content.strip():
            print(f"Warning: No text content extracted from {pdf_path}. The PDF might be scanned or image-based.")
            return []
        
        # Clean the text (remove excessive whitespace, etc.)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        try:
            # Try to split into sentences using NLTK
            sentences = sent_tokenize(text_content)
        except Exception as e:
            print(f"NLTK sentence tokenization failed: {e}")
            print("Falling back to basic sentence splitting...")
            # Fall back to basic sentence splitting
            sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', text_content) if s.strip()]
        
        # Filter out very short or empty sentences (likely parsing artifacts)
        sentences = [s for s in sentences if len(s.strip()) > 10]
        
        return sentences
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return []

def process_all_documents():
    """Process all documents in the directory and create a master JSON file."""
    # Try to load existing sentences JSON if it exists
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as json_file:
                all_docs = json.load(json_file)
                print(f"Loaded existing sentences from {OUTPUT_JSON}")
                return all_docs
        except Exception as e:
            print(f"Error loading existing sentences file: {e}. Will create a new one.")
    
    all_docs = {}
    
    # Check if directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Directory {DOCUMENTS_DIR} not found!")
        return all_docs
    
    # Walk through all files in the directory
    file_count = 0
    for root, _, files in os.walk(DOCUMENTS_DIR):
        file_count += sum(1 for file in files if file.lower().endswith('.pdf'))
    
    print(f"Found {file_count} PDF documents to process")
    
    with tqdm(total=file_count, desc="Processing documents") as pbar:
        for root, _, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    pbar.set_description(f"Processing {file}")
                    
                    # Extract sentences from PDF
                    sentences = extract_sentences(file_path)
                    print(f"Extracted {len(sentences)} sentences from {file}")
                    
                    all_docs[file] = {
                        "title": file,
                        "sentences": sentences
                    }
                    
                    pbar.update(1)
    
    # Save to JSON file
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as json_file:
        json.dump(all_docs, json_file, indent=2)
    
    print(f"Saved all sentences to {OUTPUT_JSON}")
    return all_docs

def load_requirements():
    """Load requirements from the Excel file, filtering for rows with 'red' in the Rating column."""
    try:
        df = pd.read_excel(EXCEL_PATH)
        
        # Check if required columns exist
        if "SOW Relevance" not in df.columns:
            print("Column 'SOW Relevance' not found in Excel file!")
            return None
            
        if "Rating" not in df.columns:
            print("Column 'Rating' not found in Excel file! Proceeding without filtering.")
        else:
            # Filter to only include rows with "red" rating
            red_rows = df[df["Rating"].astype(str).str.lower() == "red"]
            print(f"Filtered from {len(df)} total requirements to {len(red_rows)} 'red' rated requirements")
            df = red_rows
            
            if len(df) == 0:
                print("Warning: No rows with 'red' rating found. Check your Excel data.")
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def find_related_sentences(requirement, all_sentences, all_titles):
    """Find the top three most related sentences using sentence transformers."""
    # Initialize model (only once)
    if not hasattr(find_related_sentences, "model"):
        print("Loading sentence transformer model...")
        find_related_sentences.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    
    model = find_related_sentences.model
    
    # Encode the requirement
    req_embedding = model.encode([requirement])[0]
    
    # Use cache for embeddings if available
    if not hasattr(find_related_sentences, "embeddings_cache"):
        print("Creating sentence embeddings cache...")
        # Prepare a list of all sentences with their document titles
        all_sentence_list = []
        doc_titles = []
        
        for doc_title in all_titles:
            for sentence in all_sentences[doc_title]["sentences"]:
                all_sentence_list.append(sentence)
                doc_titles.append(doc_title)
        
        # Encode all sentences with progress bar
        print(f"Encoding {len(all_sentence_list)} sentences (this may take a while)...")
        sentence_embeddings = []
        
        batch_size = 32
        for i in tqdm(range(0, len(all_sentence_list), batch_size), desc="Encoding sentences"):
            batch = all_sentence_list[i:i+batch_size]
            batch_embeddings = model.encode(batch)
            sentence_embeddings.extend(batch_embeddings)
            
            # Add a small delay to prevent overloading
            time.sleep(0.01)
        
        # Store in cache
        find_related_sentences.embeddings_cache = {
            "sentence_list": all_sentence_list,
            "doc_titles": doc_titles,
            "embeddings": np.array(sentence_embeddings)
        }
        print("Embedding cache created!")
    
    # Calculate similarity using cached embeddings
    similarities = cosine_similarity(
        [req_embedding], 
        find_related_sentences.embeddings_cache["embeddings"]
    )[0]
    
    # Get indices of top 3 most similar sentences
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    # Get the top sentences and their document titles
    top_matches = [
        {
            "sentence": find_related_sentences.embeddings_cache["sentence_list"][idx],
            "document": find_related_sentences.embeddings_cache["doc_titles"][idx],
            "similarity_score": float(similarities[idx])
        }
        for idx in top_indices
    ]
    
    return top_matches

def assess_relevance(requirement, matches):
    """Use OpenAI to assess the relevance of matches to the requirement."""
    requirement_text = requirement
    
    # Prepare the matches text
    matches_text = ""
    for i, match in enumerate(matches, 1):
        matches_text += f"Match {i} (from document '{match['document']}'):\n{match['sentence']}\n\n"
    
    # Prepare the prompt
    prompt = f"""
    Assess the relevance of the following past performance examples to the given requirement:
    
    Requirement: {requirement_text}
    
    Past Performance Examples:
    {matches_text}
    
    Rate the relevance on the following scale:
    0 - No relevant experience
    1 - Some/Indirect Experience (similar work but not for CMS/CMMI, or work for CMS/CMMI but tangential to the RMADA requirements)
    2 - Significant Relevant Experience
    
    Please consider the document titles to determine if the work was performed for CMMI. If the title doesn't clearly indicate CMS/CMMI work, be careful about assigning a level 1 rating.
    
    Provide your rating (0, 1, or 2) with a brief explanation.
    """
    
    # Add retry mechanism for API calls
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Query OpenAI using the new client format
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating past performance relevance to requirements for government RFIs."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"OpenAI API error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to get response from OpenAI after {max_retries} attempts: {e}")
                return f"Error: Could not assess relevance due to API issues. Error: {str(e)}"

def load_checkpoint():
    """Load the checkpoint file if it exists."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as file:
                checkpoint = json.load(file)
                print(f"Checkpoint loaded, resuming from requirement {checkpoint['last_processed_index'] + 1}")
                # Check if we need to reset checkpoint when switching modes
                if MODE == "test" and checkpoint.get("mode", MODE) == "prod":
                    print("WARNING: Switching from prod to test mode with existing checkpoint.")
                    print("Consider deleting the checkpoint file if you want to start a fresh test run.")
                return checkpoint
        return {"results": [], "last_processed_index": -1, "mode": MODE}
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {"results": [], "last_processed_index": -1, "mode": MODE}

def save_checkpoint(results, last_idx):
    """Save a checkpoint with current progress."""
    checkpoint = {
        "results": results,
        "last_processed_index": last_idx,
        "timestamp": pd.Timestamp.now().isoformat(),
        "mode": MODE
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as file:
        json.dump(checkpoint, file, indent=2)
    print(f"Checkpoint saved after processing requirement {last_idx + 1}")

def main():
    print(f"Running in {MODE.upper()} mode")
    
    # Process all documents and create the master JSON
    print("Processing all documents...")
    all_docs = process_all_documents()
    
    if not all_docs:
        print("No documents processed. Exiting.")
        return
    
    # Load requirements from Excel
    print("Loading requirements from Excel...")
    requirements_df = load_requirements()
    
    if requirements_df is None:
        print("Failed to load requirements. Exiting.")
        return
    
    if len(requirements_df) == 0:
        print("No requirements to process after filtering. Exiting.")
        return
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    results = checkpoint["results"]
    start_idx = checkpoint["last_processed_index"] + 1
    
    # Limit processing if in test mode
    if MODE == "test":
        end_idx = min(10, len(requirements_df))
        if start_idx >= end_idx:
            print(f"Test mode: Already processed {start_idx} requirements, which is >= the test limit of {end_idx}.")
            print("To continue, either set MODE to 'prod' or delete the checkpoint file.")
            return
        print(f"Test mode: Will process requirements from index {start_idx} to {end_idx-1}")
        requirements_to_process = requirements_df.iloc[start_idx:end_idx]
    else:  # prod mode
        requirements_to_process = requirements_df.iloc[start_idx:]
        print(f"Production mode: Will process all remaining requirements from index {start_idx} to {len(requirements_df)-1}")
    
    # Process each requirement
    try:
        for idx, row in requirements_to_process.iterrows():
            requirement = row["SOW Relevance"]
            print(f"Processing requirement {idx+1}/{len(requirements_df)}: {requirement[:50]}...")
            
            # Find related sentences
            matches = find_related_sentences(requirement, all_docs, list(all_docs.keys()))
            
            # Assess relevance using OpenAI
            assessment = assess_relevance(requirement, matches)
            
            # Store results
            result = {
                "requirement": requirement,
                "matches": matches,
                "assessment": assessment,
                "excel_row_index": idx  # Store the original Excel row index for reference
            }
            results.append(result)
            
            # Save checkpoint after each requirement
            save_checkpoint(results, idx)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\nAn error occurred: {e}. Progress has been saved.")
    finally:
        # Save final results to JSON
        output_file = f"test_{RESULTS_JSON}" if MODE == "test" else RESULTS_JSON
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # If process completed successfully in prod mode, remove checkpoint
        if MODE == "prod" and start_idx + len(results) >= len(requirements_df):
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                print("Checkpoint file removed after successful completion.")
        elif MODE == "test":
            print("Test mode completed. Set MODE to 'prod' to process all requirements.")

def install_requirements():
    """Check and install required packages if necessary."""
    try:
        import pip
        required_packages = [
            'pandas', 
            'nltk', 
            'openai', 
            'sentence-transformers', 
            'scikit-learn',
            'tqdm',
            'PyPDF2',
            'openpyxl'  # For reading Excel files
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"Installing required package: {package}")
                pip.main(['install', package])
                
        # Download NLTK data - adding all potentially needed resources
        import nltk
        nltk_resources = ['punkt', 'punkt_tab']
        for resource in nltk_resources:
            try:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=False)
            except Exception as e:
                print(f"Error downloading NLTK resource {resource}: {e}")
                print("This might be a naming issue. Continuing with available resources.")
        
        print("All required packages are installed.")
        
        # Remind about OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nIMPORTANT: OpenAI API key not found in environment variables.")
            print("Please set your API key using one of these methods:")
            print("1. Set the OPENAI_API_KEY environment variable")
            print("2. Or add the following line after the client initialization in the script:")
            print("   client.api_key = \"your-api-key-here\"")
    except Exception as e:
        print(f"Error checking/installing requirements: {e}")
        print("Please manually install the required packages:")
        print("pip install pandas nltk openai sentence-transformers scikit-learn tqdm PyPDF2 openpyxl")
        print("And download NLTK data: python -m nltk.downloader punkt")
        exit(1)

if __name__ == "__main__":
    # Check requirements
    install_requirements()
    
    # Allow command line override of mode
    import argparse
    
    parser = argparse.ArgumentParser(description='RFI Document Analysis Script')
    parser.add_argument('--mode', choices=['test', 'prod'], 
                      help='Run mode: test (first 10 requirements) or prod (all requirements)')
    args = parser.parse_args()
    
    if args.mode:
        MODE = args.mode
        print(f"Mode set from command line: {MODE}")
        
    main()