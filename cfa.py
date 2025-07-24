import json
import logging

logger = logging.getLogger(__name__)

# CFA Specialties list (can be moved to config file later if needed)
CFA_SPECIALTIES = """
Addiction medicine, Allergy and clinical immunology, Anesthesia, Cardiac surgery, Cardiology, Chronic pain, Critical care medicine, Dermatology, 
Diagnostic and interventional radiology, Emergency medicine, Endocrinology, Gastroenterology, General and family practice, General internal medicine, 
General surgery, General thoracic surgery, Genetics medicine, Geriatric medicine, Hematology and medical oncology, Hospital medicine, 
Infectious diseases, Laboratory medicine, Long-term care/care of the elderly, Nephrology, Neurology, Neuroradiology, Neurosurgery, Nuclear medicine, 
Obstetrics and gynecology, Occupational and environmental medicine, Opthalmology, Orthopedic surgery, Otolaryngology/head and neck surgery, 
Palliative medicine, Pediatrics, Physical medicine and rehabilitation, Plastic surgery, Primary care mental health, Psychiatry, 
Public health physicians, Radiation oncology, Reproductive biology, Respiratory disease, Rheumatology, Sport and exercise medicine, Urology, 
Vascular surgery
"""

def create_diagnosis_prompt(priority_terms):
    """
    Create a prompt for predicting diagnoses based on priority clinical terms
    
    Args:
        priority_terms (list): List of priority clinical terms (canonical names)
        
    Returns:
        str: Formatted prompt for Claude AI
    """
    if not priority_terms:
        return None
    
    terms_text = ', '.join(priority_terms)
    
    return f"""
You are a medical AI assistant.
Based on the following clinical terms extracted from medical notes, provide any findings or diagnoses.

Clinical Terms: {terms_text}

Please provide only the diagnosis names (at most 5, can be fewer), one per line, without explanations or additional text.

Example format:
Strep Throat
Dental filling lost
"""

def create_cfa_prediction_prompt(selected_diagnoses):
    """
    Create a prompt for predicting CFA specialties based on selected diagnoses
    
    Args:
        selected_diagnoses (list): List of selected diagnoses
        
    Returns:
        str: Formatted prompt for Claude AI
    """
    if not selected_diagnoses:
        return None
    
    diagnoses_text = '\n'.join(selected_diagnoses)
    
    return f"""
You are a medical AI assistant. For each diagnosis below, predict the most appropriate medical specialty from the provided list.

Diagnoses:
{diagnoses_text}

Available Medical Specialties:
{CFA_SPECIALTIES}

For each diagnosis, return ONLY the specialty name from the provided list that would most likely handle that condition.
IMPORTANT: ONLY one specialty per diagnosis.
Format your response like the following example, ONLY including the diagnosis and its corresponding specialty:

Dental caries: Restorative Dentistry
Prediabetes: Chronic Disease Management
Pregnancy: Prenatal Care
"""

def predict_diagnoses(bedrock_client, model_id, priority_terms, max_tokens=4000):
    """
    Predict diagnoses based on priority clinical terms using Claude AI
    
    Args:
        bedrock_client: Initialized boto3 bedrock client
        model_id (str): Claude model ID for bedrock
        priority_terms (list): List of priority clinical terms
        max_tokens (int): Maximum tokens for the response
        
    Returns:
        list: List of predicted diagnoses (max 5)
        
    Raises:
        Exception: If the diagnosis prediction fails
    """
    try:
        logger.info(f"Starting diagnosis prediction with {len(priority_terms)} priority terms")
        
        # Validate inputs
        if not priority_terms:
            logger.warning("No priority terms provided for diagnosis prediction")
            return []
        
        if not bedrock_client:
            raise ValueError("Bedrock client is required")
        
        # Create the diagnosis prompt
        prompt = create_diagnosis_prompt(priority_terms)
        if not prompt:
            logger.warning("Could not create diagnosis prompt")
            return []
        
        # Prepare request body for bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        logger.info("Sending diagnosis prediction request to Claude AI via Bedrock")
        
        # Call bedrock
        response = bedrock_client.invoke_model(
            modelId=model_id, 
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        diagnosis_response = response_body['content'][0]['text']
        
        # Parse diagnoses (split by lines and clean)
        diagnoses = [d.strip() for d in diagnosis_response.split('\n') if d.strip()]
        diagnoses = diagnoses[:5]  # Ensure max 5 diagnoses
        
        logger.info(f"Successfully predicted {len(diagnoses)} diagnoses")
        return diagnoses
        
    except Exception as e:
        logger.error(f"Error predicting diagnoses: {str(e)}")
        raise Exception(f"Failed to predict diagnoses: {str(e)}")

def predict_cfa_specialties(bedrock_client, model_id, selected_diagnoses, max_tokens=4000):
    """
    Predict CFA specialties based on selected diagnoses using Claude AI
    
    Args:
        bedrock_client: Initialized boto3 bedrock client
        model_id (str): Claude model ID for bedrock
        selected_diagnoses (list): List of selected diagnoses
        max_tokens (int): Maximum tokens for the response
        
    Returns:
        dict: Dictionary mapping diagnoses to predicted specialties
        
    Raises:
        Exception: If the CFA prediction fails
    """
    try:
        logger.info(f"Starting CFA specialty prediction for {len(selected_diagnoses)} diagnoses")
        
        # Validate inputs
        if not selected_diagnoses:
            logger.warning("No selected diagnoses provided for CFA prediction")
            return {}
        
        if not bedrock_client:
            raise ValueError("Bedrock client is required")
        
        # Create the CFA prediction prompt
        prompt = create_cfa_prediction_prompt(selected_diagnoses)
        if not prompt:
            logger.warning("Could not create CFA prediction prompt")
            return {}
        
        # Prepare request body for bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        logger.info("Sending CFA prediction request to Claude AI via Bedrock")
        
        # Call bedrock
        response = bedrock_client.invoke_model(
            modelId=model_id, 
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        cfa_response = response_body['content'][0]['text']
        
        # Parse CFA results
        cfa_predictions = {}
        for line in cfa_response.split('\n'):
            if ':' in line:
                diagnosis, specialty = line.split(':', 1)
                cfa_predictions[diagnosis.strip()] = specialty.strip()
        
        logger.info(f"Successfully predicted CFA specialties for {len(cfa_predictions)} diagnoses")
        return cfa_predictions
        
    except Exception as e:
        logger.error(f"Error predicting CFA specialties: {str(e)}")
        raise Exception(f"Failed to predict CFA specialties: {str(e)}")

def validate_diagnoses(diagnoses):
    """
    Validate predicted diagnoses
    
    Args:
        diagnoses (list): List of predicted diagnoses
        
    Returns:
        dict: Validation results with 'valid' boolean and 'issues' list
    """
    issues = []
    
    if not diagnoses:
        issues.append("No diagnoses predicted")
    
    if len(diagnoses) > 5:
        issues.append("Too many diagnoses (max 5 allowed)")
    
    # Check for empty or very short diagnoses
    for i, diagnosis in enumerate(diagnoses):
        if not diagnosis or len(diagnosis.strip()) < 3:
            issues.append(f"Diagnosis {i+1} is too short or empty")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'count': len(diagnoses)
    }

def validate_cfa_predictions(cfa_predictions, selected_diagnoses):
    """
    Validate CFA specialty predictions
    
    Args:
        cfa_predictions (dict): Dictionary of diagnosis -> specialty mappings
        selected_diagnoses (list): List of diagnoses that should have predictions
        
    Returns:
        dict: Validation results with 'valid' boolean and 'issues' list
    """
    issues = []
    
    if not cfa_predictions:
        issues.append("No CFA specialties predicted")
    
    # Check if all selected diagnoses have predictions
    missing_predictions = []
    for diagnosis in selected_diagnoses:
        if diagnosis not in cfa_predictions:
            missing_predictions.append(diagnosis)
    
    if missing_predictions:
        issues.append(f"Missing CFA predictions for: {', '.join(missing_predictions)}")
    
    # Check for empty specialties
    empty_specialties = []
    for diagnosis, specialty in cfa_predictions.items():
        if not specialty or len(specialty.strip()) < 3:
            empty_specialties.append(diagnosis)
    
    if empty_specialties:
        issues.append(f"Empty or invalid specialties for: {', '.join(empty_specialties)}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'predictions_count': len(cfa_predictions),
        'coverage': len(cfa_predictions) / len(selected_diagnoses) if selected_diagnoses else 0
    }