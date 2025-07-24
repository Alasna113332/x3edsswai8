import json
import logging

logger = logging.getLogger(__name__)

def create_soap_cleaning_prompt(soap_notes, location):
    """
    Create a structured prompt for cleaning SOAP notes using Claude AI
    
    Args:
        soap_notes (str): Raw doctor's notes to be cleaned
        location (str): Location context for the medical notes
        
    Returns:
        str: Formatted prompt for Claude AI
    """
    return f"""
You are a medical AI assistant specialized in processing doctor's notes. Your task is to clean the notes while maintaining medical accuracy.

Location: {location}

Doctor's Notes to process:
{soap_notes} 

Extract and generate 2 concise sentences that capture all of the medical details from the notes. Also handle the following:
1. Disambiguate abbreviations (e.g., RA, PT, MED)
2. Fix typos and informal language (e.g., "diabtes" → "diabetes")
3. Detect and clarify negation/uncertainty ("no chest pain", "possible pneumonia")
4. Normalize synonyms (e.g., MI = heart attack = myocardial infarction)
5. Resolve pronouns (e.g., "she" = patient)
6. Expand shorthand/jargon (e.g., "+ve strep", "c/o")
7. Clarify missing/implicit info ("Started insulin" → patient started insulin)
8. Consolidate nested/overlapping terms (e.g., chronic kidney disease stage 3)
9. Handle code-switched/mixed language (e.g., "douleur abdominale")
10. Correct formatting/punctuation (e.g., "N/V/D x3d. Dx: ?appx")
11. Maintain original timeframes as given in the {soap_notes}
12. Do not calculate ages or durations
13. Indicate whether medications are prescribed, current, or historical

Output only clean, medically accurate sentences reflecting the current and historical clinical issue or diagnostic focus.

Do not diagnose based on the notes or add extra information or inference.

**Example:** 
Input: Client came in today with reports of sore throat. Symptoms include difficulty swallowing, fever and swollen tonsils. Note client has a history of dental concerns related to fillings and acute infective cystitis. History of smoking at age 20 (11 years ago), reports has not smoked since. Throat culture conducted and blood work ordered. Client aware will only be contacted in case of abnormal results. MED: penicillin v potassium 500 mg oral tablet 

Output: The patient presented with sore throat, difficulty swallowing, fever, and swollen tonsils, with throat culture conducted and bloodwork ordered. The patient has a history of dental concerns related to fillings, acute infective cystitis, and smoking at age 20 (11 years ago), and is prescribed penicillin v potassium 500 mg oral tablet with contact planned only for abnormal results.
"""

def clean_soap_notes(bedrock_client, model_id, soap_notes, location="Not specified", max_tokens=4000):
    """
    Clean SOAP notes using AWS Bedrock Claude AI
    
    Args:
        bedrock_client: Initialized boto3 bedrock client
        model_id (str): Claude model ID for bedrock
        soap_notes (str): Raw medical notes to clean
        location (str): Location context for the notes
        max_tokens (int): Maximum tokens for the response
        
    Returns:
        str: Cleaned medical notes
        
    Raises:
        Exception: If the cleaning process fails
    """
    try:
        logger.info(f"Starting SOAP notes cleaning for location: {location}")
        
        # Validate inputs
        if not soap_notes or not soap_notes.strip():
            raise ValueError("SOAP notes cannot be empty")
        
        if not bedrock_client:
            raise ValueError("Bedrock client is required")
        
        # Create the cleaning prompt
        prompt = create_soap_cleaning_prompt(soap_notes, location)
        
        # Prepare request body for bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        logger.info("Sending cleaning request to Claude AI via Bedrock")
        
        # Call bedrock
        response = bedrock_client.invoke_model(
            modelId=model_id, 
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        cleaned_text = response_body['content'][0]['text']
        
        logger.info("SOAP notes cleaned successfully")
        logger.debug(f"Original length: {len(soap_notes)}, Cleaned length: {len(cleaned_text)}")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error cleaning SOAP notes: {str(e)}")
        raise Exception(f"Failed to clean SOAP notes: {str(e)}")

def validate_cleaned_text(cleaned_text):
    """
    Validate that cleaned text meets basic requirements
    
    Args:
        cleaned_text (str): The cleaned text to validate
        
    Returns:
        dict: Validation results with 'valid' boolean and 'issues' list
    """
    issues = []
    
    if not cleaned_text or not cleaned_text.strip():
        issues.append("Cleaned text is empty")
    
    if len(cleaned_text.strip()) < 10:
        issues.append("Cleaned text is too short")
    
    # Check for basic medical content (at least some medical terms expected)
    medical_indicators = [
        'patient', 'symptoms', 'diagnosis', 'treatment', 'medication', 
        'history', 'examination', 'findings', 'condition', 'therapy'
    ]
    
    has_medical_content = any(indicator in cleaned_text.lower() for indicator in medical_indicators)
    if not has_medical_content:
        issues.append("Cleaned text may not contain medical content")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'length': len(cleaned_text),
        'has_medical_content': has_medical_content
    }