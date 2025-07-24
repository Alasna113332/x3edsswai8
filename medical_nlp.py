import os
import spacy
from scispacy.linking import EntityLinker
from negspacy.negation import Negex
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalNLPProcessor:
    def __init__(self):
        """Initialize the medical NLP pipeline"""
        self.nlp = None
        self.priority_semtypes = {
            "T033","T001", "T017", "T019", "T020", "T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030", "T031", "T032", "T034", "T037", "T038", "T039", "T040", "T041", "T042", "T043", "T044", "T045", "T046", "T047", "T048", "T049", "T050", "T052", "T056", "T057", "T058", "T059", "T060", "T061", "T062", "T064", "T065", "T066", "T067", "T068", "T069", "T070", "T074","T080","T081", "T091", "T092", "T093", "T095", "T102", "T109", "T120", "T121", "T122", "T123", "T125", "T126", "T130", "T131", "T167", "T169", "T170", "T184", "T190", "T191", "T192", "T197", "T200", "T201", "T203", "T054", "T055","T195", "T100"
        }
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP pipeline with minimal components for speed"""
        try:
            logger.info("Loading optimized spaCy medical pipeline...")
            
            # Load with only essential components
            self.nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "textcat"])
            
            # Add only negation detection (keeping this as requested)
            self.nlp.add_pipe("negex", 
                config={"chunk_prefix": ["no", "without", "denies", "not"]}, 
                last=False)
            
            # Add UMLS entity linker with faster configuration
            self.nlp.add_pipe("scispacy_linker", 
                config={
                    "resolve_abbreviations": False,  # Disabled for speed
                    "linker_name": "umls",
                    "threshold": 0.8  # Higher threshold for faster processing
                })
            
            logger.info("Optimized medical NLP pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize medical NLP: {e}")
            raise
    
    def extract_clinical_concepts(self, cleaned_text: str):
        """Extract clinical concepts with optimized processing"""
        try:
            logger.info("Processing text with optimized medical NLP...")
            doc = self.nlp(cleaned_text)
            
            cfa_list = []
            seen_cuis = set()  # Fast duplicate checking
            
            for ent in doc.ents:
                # Quick filtering - only basic checks
                if len(ent.text.strip()) < 3 or not ent._.kb_ents:
                    continue
                
                # Skip negated entities (keeping negation as requested)
                if hasattr(ent._, "negex") and ent._.negex:
                    continue
                
                # Get top entity match only
                cui, score = ent._.kb_ents[0]
                
                # Skip if we've already seen this CUI
                if cui in seen_cuis:
                    continue
                seen_cuis.add(cui)
                
                # Get entity information
                entity = self.nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui]
                
                # Quick priority check
                semantic_types = set(entity.types)
                is_priority = bool(semantic_types.intersection(self.priority_semtypes))
                
                cfa_list.append({
                    "Term": ent.text.strip(),
                    "UMLS_CUI": cui,
                    "Canonical_Name": entity.canonical_name,
                    "Score": round(score, 3),
                    "Semantic_Types": ", ".join(entity.types),
                    "Priority": is_priority
                })
            
            # Create DataFrame
            if not cfa_list:
                logger.warning("No clinical concepts found")
                return pd.DataFrame()
            
            cfa_df = pd.DataFrame(cfa_list)
            
            # Simple sorting - priority first, then score
            cfa_df = cfa_df.sort_values(['Priority', 'Score'], ascending=[False, False]).reset_index(drop=True)
            
            logger.info(f"Extracted {len(cfa_df)} unique clinical concepts")
            
            return cfa_df
            
        except Exception as e:
            logger.error(f"Error extracting clinical concepts: {e}")
            raise
    
    def get_priority_concepts(self, cfa_df):
        """Filter and return only priority concepts"""
        if cfa_df.empty:
            return pd.DataFrame()
        
        return cfa_df[cfa_df['Priority'] == True].reset_index(drop=True)
    
    def get_concept_summary(self, cfa_df):
        """Get basic summary statistics"""
        if cfa_df.empty:
            return {
                "total_concepts": 0,
                "priority_concepts": 0,
                "semantic_type_distribution": {},
                "score_range": {"min": 0, "max": 0, "mean": 0}
            }
        
        # Simplified semantic type counting
        semantic_type_counts = {}
        for semantic_types_str in cfa_df['Semantic_Types']:
            for sem_type in semantic_types_str.split(', '):
                semantic_type_counts[sem_type] = semantic_type_counts.get(sem_type, 0) + 1
        
        return {
            "total_concepts": len(cfa_df),
            "priority_concepts": len(cfa_df[cfa_df['Priority'] == True]),
            "semantic_type_distribution": semantic_type_counts,
            "score_range": {
                "min": float(cfa_df['Score'].min()),
                "max": float(cfa_df['Score'].max()),
                "mean": float(cfa_df['Score'].mean())
            }
        }

# Global instance (initialized once)
medical_processor = None

def get_medical_processor():
    """Get or create the medical processor instance"""
    global medical_processor
    if medical_processor is None:
        medical_processor = MedicalNLPProcessor()
    return medical_processor

def process_soap_for_cfa(cleaned_text: str):
    """Main function to process SOAP notes and extract CFA concepts"""
    try:
        processor = get_medical_processor()
        
        # Extract concepts as DataFrame
        concepts_df = processor.extract_clinical_concepts(cleaned_text)
        
        if concepts_df.empty:
            return {
                "concepts_dataframe": concepts_df,
                "priority_concepts_dataframe": pd.DataFrame(),
                "summary": {
                    "total_concepts": 0,
                    "priority_concepts": 0,
                    "message": "No clinical concepts found in the text"
                }
            }
        
        # Get priority concepts
        priority_df = processor.get_priority_concepts(concepts_df)
        
        # Get summary statistics
        summary = processor.get_concept_summary(concepts_df)
        summary["message"] = f"Successfully extracted {len(concepts_df)} clinical concepts"
        
        return {
            "concepts_dataframe": concepts_df,
            "priority_concepts_dataframe": priority_df,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error in process_soap_for_cfa: {e}")
        raise

def extract_priority_terms_list(priority_concepts_df):
    """
    Extract priority terms as a simple list from DataFrame
    
    Args:
        priority_concepts_df: DataFrame with priority concepts
        
    Returns:
        list: List of priority terms (canonical names)
    """
    try:
        if priority_concepts_df.empty:
            logger.info("No priority concepts found")
            return []
        
        # Extract canonical names as the priority terms
        priority_terms = priority_concepts_df['Canonical_Name'].tolist()
        
        logger.info(f"Extracted {len(priority_terms)} priority terms")
        return priority_terms
        
    except Exception as e:
        logger.error(f"Error extracting priority terms list: {e}")
        return []
    