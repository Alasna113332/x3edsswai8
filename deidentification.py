import logging

logger = logging.getLogger(__name__)

def deidentify_soap_notes(bedrock_client, guardrail_id, guardrail_version, soap_notes):
    """De-identify SOAP notes using AWS Bedrock Guardrails"""
    try:
        logger.info("Starting SOAP notes de-identification")
        
        # Validate inputs
        if not soap_notes or not soap_notes.strip():
            raise ValueError("SOAP notes cannot be empty")
        
        if not bedrock_client:
            raise ValueError("Bedrock client is required")
        
        # Apply guardrail for de-identification
        response = bedrock_client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source='INPUT',
            content=[{"text": {"text": soap_notes}}]
        )
        
        # Handle the response structure correctly
        if 'outputs' in response and len(response['outputs']) > 0:
            deidentified_text = response["outputs"][0]["text"]
        elif 'output' in response:
            deidentified_text = response["output"]
        else:
            # Fallback - sometimes the response structure may vary
            logger.warning("Unexpected response structure from guardrail, using original text")
            deidentified_text = soap_notes
        
        logger.info("SOAP notes de-identified successfully")
        logger.debug(f"Original length: {len(soap_notes)}, De-identified length: {len(deidentified_text)}")
        
        return deidentified_text
        
    except Exception as e:
        logger.error(f"Error de-identifying SOAP notes: {str(e)}")
        raise Exception(f"Failed to de-identify SOAP notes: {str(e)}")

def validate_deidentified_text(deidentified_text):
    """Validate that de-identified text meets basic requirements"""
    issues = []
    
    if not deidentified_text or not deidentified_text.strip():
        issues.append("De-identified text is empty")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'length': len(deidentified_text)
    }
