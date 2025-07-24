import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import numpy as np
from typing import List
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model variables (loaded once)
tokenizer = None
model = None
device = torch.device('cpu')

def load_model():
    """Load Bio-ClinicalBERT model and tokenizer"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        logger.info("Loading Bio-ClinicalBERT model...")
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")

def create_embeddings(texts: List[str]) -> np.ndarray:
    """
    Create normalized embeddings for a list of text strings using Bio-ClinicalBERT.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        np.ndarray: Normalized embeddings with shape (batch_size, hidden_size)
        
    Raises:
        ValueError: If input list is empty or contains no valid texts
    """
    if not texts:
        raise ValueError("Input list is empty")
    
    # Ensure model is loaded
    load_model()
    
    # Clean and filter empty texts
    cleaned_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not cleaned_texts:
        raise ValueError("No valid (non-empty) texts after cleaning")
    
    # Tokenize as batch
    inputs = tokenizer(
        cleaned_texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Create mean pooled embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def load_terms_from_excel(file_path: str) -> List[str]:
    """
    Load medical specialty terms from Excel file.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        List[str]: Cleaned list of terms from 'Secondary_Category' column
    """
    try:
        logger.info(f"Loading Excel file: {file_path}")
        df = pd.read_excel(file_path)

        # Extract and clean terms from the 'Secondary_Category' column
        if "Secondary_Category" not in df.columns:
            raise ValueError("Column 'Secondary_Category' not found in Excel file.")
        
        terms = df["Secondary_Category"].dropna().astype(str).tolist()
        terms = [term.strip() for term in terms if term.strip()]
        
        logger.info(f"Loaded {len(terms)} valid terms")
        return terms
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def setup_chromadb(db_path: str = "./chroma_db") -> chromadb.Collection:
    """
    Set up ChromaDB client and collection.
    
    Args:
        db_path: Path to ChromaDB storage directory
        
    Returns:
        chromadb.Collection: The CFA terms collection
    """
    logger.info(f"Setting up ChromaDB at: {db_path}")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="cfa_terms")
        logger.info("Deleted existing collection")
    except:
        pass  # Collection doesn't exist
    
    # Create new collection
    collection = client.create_collection(
        name="cfa_terms",
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info("Created new ChromaDB collection")
    return collection

def create_and_store_embeddings(terms: List[str], collection: chromadb.Collection):
    """
    Create embeddings for terms and store them in ChromaDB.
    
    Args:
        terms: List of medical specialty terms
        collection: ChromaDB collection to store embeddings
    """
    logger.info("Creating embeddings and storing in database...")
    
    documents, embeddings, ids, metadatas = [], [], [], []
    batch_size = 10  # Process in smaller batches to avoid memory issues
    
    for i in range(0, len(terms), batch_size):
        batch_terms = terms[i:i + batch_size]
        batch_start_idx = i
        
        try:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(terms)-1)//batch_size + 1}: terms {i+1}-{min(i+batch_size, len(terms))}")
            
            # Create embeddings for batch
            batch_embeddings = create_embeddings(batch_terms)
            
            # Process each term in the batch
            for j, (term, embedding) in enumerate(zip(batch_terms, batch_embeddings)):
                term_idx = batch_start_idx + j
                
                documents.append(term)
                embeddings.append(embedding.tolist())  # Convert to list for ChromaDB
                ids.append(f"term_{term_idx}")
                
                metadata = {
                    "original_term": term,
                    "term_length": len(term),
                    "word_count": len(term.split()),
                    "index": term_idx
                }
                metadatas.append(metadata)
                
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")
            continue
    
    # Store in ChromaDB
    if documents:
        logger.info(f"Storing {len(documents)} embeddings in ChromaDB...")
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully stored {len(documents)} terms in database!")
        logger.info(f"Database now contains {collection.count()} total terms")
    else:
        logger.error("No embeddings were created successfully!")

def main():
    """Main function to set up the ChromaDB with CFA terms"""
    # Configuration
    excel_file_path = r"C:/Users/Abhi/Downloads/Book1.xlsx"
    db_path = "./chroma_db"
    
    try:
        # Check if Excel file exists
        if not os.path.exists(excel_file_path):
            logger.error(f"Excel file not found: {excel_file_path}")
            logger.info("Please update the excel_file_path variable with the correct path to your Excel file")
            return
        
        # Load terms from Excel
        terms = load_terms_from_excel(excel_file_path)
        
        if not terms:
            logger.error("No terms loaded from Excel file")
            return
        
        # Set up ChromaDB
        collection = setup_chromadb(db_path)
        
        # Create and store embeddings
        create_and_store_embeddings(terms, collection)
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()