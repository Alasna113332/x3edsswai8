from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import boto3
import json
import os
import threading
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import your modules
from medical_nlp import process_soap_for_cfa, get_medical_processor, extract_priority_terms_list
from cleaning import clean_soap_notes, validate_cleaned_text
from cfa import predict_diagnoses, predict_cfa_specialties, validate_diagnoses, validate_cfa_predictions
from deidentification import deidentify_soap_notes, validate_deidentified_text
from logic import (
    predict_cfa_from_priority_terms, 
    create_visualization_data, 
    create_prediction_chart_data,
    get_selectable_cfa_list,
    get_cfa_engine
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"], supports_credentials=True)

# AWS Configuration - Use your existing credentials
session = boto3.Session(profile_name="PowerUserAccess-148761655243")
bedrock_client = session.client("bedrock-runtime", region_name="ca-central-1")
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Guardrail configuration for de-identification
GUARDRAIL_ID = 'wym574vww1er'
GUARDRAIL_VERSION = '2'

# Global variables for shared resources
medical_processor = None
cfa_engine = None
pipeline_storage = {}

# Lock for thread-safe operations
storage_lock = threading.Lock()

# Initialize medical processor on startup
print("Initializing medical NLP processor...")
try:
    medical_processor = get_medical_processor()
    print("Medical NLP processor ready!")
except Exception as e:
    print(f"Warning: Medical NLP processor initialization failed: {e}")
    medical_processor = None

# Initialize CFA engine on startup
print("Initializing CFA prediction engine...")
try:
    cfa_engine = get_cfa_engine()
    print("CFA prediction engine ready!")
except Exception as e:
    print(f"Warning: CFA engine initialization failed: {e}")
    cfa_engine = None

# Background processing functions - matching your original structure
def process_pipeline_background(pipeline_id, soap_notes, location):
    """Background processing pipeline for de-identification, cleaning and term extraction"""
    try:
        print(f"Starting pipeline {pipeline_id}")
        
        # Update status to de-identifying
        pipeline_storage[pipeline_id]['status'] = 'deidentifying'
        pipeline_storage[pipeline_id]['stage'] = 'De-identifying SOAP notes...'
        
        # Step 1: De-identify the notes using guardrails
        deidentified_text = deidentify_soap_notes(
            bedrock_client=bedrock_client,
            guardrail_id=GUARDRAIL_ID,
            guardrail_version=GUARDRAIL_VERSION,
            soap_notes=soap_notes
        )
        
        # Validate de-identified text
        validation_result = validate_deidentified_text(deidentified_text)
        if not validation_result['valid']:
            raise Exception(f"De-identified text validation failed: {', '.join(validation_result['issues'])}")
        
        # Store de-identified text
        pipeline_storage[pipeline_id]['deidentified_text'] = deidentified_text
        pipeline_storage[pipeline_id]['status'] = 'cleaning'
        pipeline_storage[pipeline_id]['stage'] = 'Cleaning notes with Claude AI...'
        
        print(f"Pipeline {pipeline_id}: Notes de-identified, starting cleaning")
        
        # Step 2: Clean the de-identified notes using the cleaning module
        cleaned_text = clean_soap_notes(
            bedrock_client=bedrock_client,
            model_id=CLAUDE_MODEL_ID,
            soap_notes=deidentified_text,  # Use de-identified text for cleaning
            location=location
        )
        
        # Validate cleaned text
        validation_result = validate_cleaned_text(cleaned_text)
        if not validation_result['valid']:
            raise Exception(f"Cleaned text validation failed: {', '.join(validation_result['issues'])}")
        
        # Store cleaned text
        pipeline_storage[pipeline_id]['cleaned_text'] = cleaned_text
        pipeline_storage[pipeline_id]['status'] = 'extracting'
        pipeline_storage[pipeline_id]['stage'] = 'Extracting clinical concepts with spaCy...'
        
        print(f"Pipeline {pipeline_id}: Notes cleaned, starting term extraction")
        
        # Step 3: Extract clinical concepts
        if medical_processor is None:
            raise Exception("Medical NLP processor not available")
            
        cfa_result = process_soap_for_cfa(cleaned_text)
        
        # Check if extraction was successful
        if cfa_result['concepts_dataframe'].empty:
            print(f"Pipeline {pipeline_id}: No clinical concepts extracted")
            concepts_data = {
                'all_concepts': [],
                'priority_concepts': [],
                'priority_terms': [],
                'summary': cfa_result['summary']
            }
        else:
            # Step 4: Extract priority terms as a list
            priority_terms = extract_priority_terms_list(cfa_result['priority_concepts_dataframe'])
            
            concepts_data = {
                'all_concepts': cfa_result['concepts_dataframe'].to_dict('records'),
                'priority_concepts': cfa_result['priority_concepts_dataframe'].to_dict('records'),
                'priority_terms': priority_terms,
                'summary': cfa_result['summary']
            }
            
            print(f"Pipeline {pipeline_id}: Extracted {len(priority_terms)} priority terms from {len(cfa_result['concepts_dataframe'])} concepts")
        
        # Store extraction results
        pipeline_storage[pipeline_id]['concepts_data'] = concepts_data
        pipeline_storage[pipeline_id]['status'] = 'completed'
        pipeline_storage[pipeline_id]['stage'] = 'Processing completed!'
        
        print(f"Pipeline {pipeline_id}: Completed successfully")
        
    except Exception as e:
        print(f"Pipeline {pipeline_id} error: {e}")
        pipeline_storage[pipeline_id]['status'] = 'error'
        pipeline_storage[pipeline_id]['stage'] = f'Error: {str(e)}'
        pipeline_storage[pipeline_id]['error'] = str(e)

def process_diagnosis_background(pipeline_id):
    """Background processing for diagnosis prediction using CFA module"""
    try:
        print(f"Starting diagnosis prediction for pipeline {pipeline_id}")
        
        pipeline_data = pipeline_storage[pipeline_id]
        priority_terms = pipeline_data['concepts_data']['priority_terms']
        
        if not priority_terms:
            raise Exception("No priority terms available for diagnosis prediction")
        
        pipeline_storage[pipeline_id]['diagnosis_status'] = 'processing'
        pipeline_storage[pipeline_id]['diagnosis_stage'] = 'Predicting diagnoses with Claude AI...'
        
        # Get diagnosis predictions using CFA module
        diagnoses = predict_diagnoses(
            bedrock_client=bedrock_client,
            model_id=CLAUDE_MODEL_ID,
            priority_terms=priority_terms
        )
        
        # Validate diagnoses
        validation_result = validate_diagnoses(diagnoses)
        if not validation_result['valid']:
            print(f"Warning: Diagnosis validation issues: {', '.join(validation_result['issues'])}")
        
        pipeline_storage[pipeline_id]['diagnoses'] = diagnoses
        pipeline_storage[pipeline_id]['diagnosis_status'] = 'completed'
        pipeline_storage[pipeline_id]['diagnosis_stage'] = 'Diagnosis prediction completed!'
        
        print(f"Pipeline {pipeline_id}: Diagnosis prediction completed with {len(diagnoses)} diagnoses")
        
    except Exception as e:
        print(f"Pipeline {pipeline_id} diagnosis error: {e}")
        pipeline_storage[pipeline_id]['diagnosis_status'] = 'error'
        pipeline_storage[pipeline_id]['diagnosis_stage'] = f'Error: {str(e)}'
        pipeline_storage[pipeline_id]['diagnosis_error'] = str(e)

def process_similarity_cfa_background(pipeline_id):
    """Background processing for similarity-based CFA prediction using Bio-ClinicalBERT"""
    try:
        print(f"Starting similarity-based CFA prediction for pipeline {pipeline_id}")
        
        pipeline_data = pipeline_storage[pipeline_id]
        
        if pipeline_data.get('concepts_data') is None:
            raise Exception('Clinical terms not available')
        
        priority_terms = pipeline_data['concepts_data'].get('priority_terms', [])
        if not priority_terms:
            raise Exception('No priority terms available for similarity-based CFA prediction')
        
        # Initialize similarity CFA prediction status
        pipeline_storage[pipeline_id]['similarity_cfa_status'] = 'processing'
        pipeline_storage[pipeline_id]['similarity_cfa_stage'] = 'Calculating similarity scores with Bio-ClinicalBERT...'
        
        # Run similarity-based prediction using logic.py
        cfa_results = predict_cfa_from_priority_terms(priority_terms)
        
        if "error" in cfa_results:
            raise Exception(cfa_results["error"])
        
        # Store results
        pipeline_storage[pipeline_id]['similarity_cfa_predictions'] = cfa_results
        pipeline_storage[pipeline_id]['similarity_cfa_status'] = 'completed'
        pipeline_storage[pipeline_id]['similarity_cfa_stage'] = 'Similarity-based CFA prediction completed!'
        
        print(f"Pipeline {pipeline_id}: Similarity CFA prediction completed with {len(cfa_results.get('cfa_predictions', []))} predictions")
        
    except Exception as e:
        print(f"Pipeline {pipeline_id} similarity CFA error: {e}")
        pipeline_storage[pipeline_id]['similarity_cfa_status'] = 'error'
        pipeline_storage[pipeline_id]['similarity_cfa_stage'] = f'Error: {str(e)}'
        pipeline_storage[pipeline_id]['similarity_cfa_error'] = str(e)

def process_cfa_background(pipeline_id, selected_diagnoses):
    """Background processing for CFA prediction using CFA module"""
    try:
        print(f"Starting CFA prediction for pipeline {pipeline_id}")
        
        pipeline_storage[pipeline_id]['cfa_status'] = 'processing'
        pipeline_storage[pipeline_id]['cfa_stage'] = 'Predicting medical specialties with Claude AI...'
        
        # Get CFA predictions using CFA module
        cfa_predictions = predict_cfa_specialties(
            bedrock_client=bedrock_client,
            model_id=CLAUDE_MODEL_ID,
            selected_diagnoses=selected_diagnoses
        )
        
        # Validate CFA predictions
        validation_result = validate_cfa_predictions(cfa_predictions, selected_diagnoses)
        if not validation_result['valid']:
            print(f"Warning: CFA validation issues: {', '.join(validation_result['issues'])}")
        
        pipeline_storage[pipeline_id]['cfa_predictions'] = cfa_predictions
        pipeline_storage[pipeline_id]['cfa_status'] = 'completed'
        pipeline_storage[pipeline_id]['cfa_stage'] = 'CFA prediction completed!'
        
        print(f"Pipeline {pipeline_id}: CFA prediction completed with {len(cfa_predictions)} predictions")
        
    except Exception as e:
        print(f"Pipeline {pipeline_id} CFA error: {e}")
        pipeline_storage[pipeline_id]['cfa_status'] = 'error'
        pipeline_storage[pipeline_id]['cfa_stage'] = f'Error: {str(e)}'
        pipeline_storage[pipeline_id]['cfa_error'] = str(e)

# Flask Routes - Static file serving
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Core Pipeline Endpoints - matching your frontend exactly
@app.route('/api/start-pipeline', methods=['POST', 'OPTIONS'])
def start_pipeline():
    """Start the background processing pipeline"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    data = request.get_json()
    soap_notes = data.get('soapNotes', '').strip()
    location = data.get('location', '').strip()
    
    if not soap_notes:
        return jsonify({'error': 'SOAP notes are required'}), 400
    
    # Generate unique pipeline ID
    pipeline_id = str(uuid.uuid4())
    
    # Initialize pipeline storage - matching your original structure
    pipeline_storage[pipeline_id] = {
        'status': 'starting',
        'stage': 'Initializing pipeline...',
        'soap_notes': soap_notes,
        'location': location or 'Not specified',
        'deidentified_text': None,
        'cleaned_text': None,
        'concepts_data': None,
        'error': None
    }
    
    # Start background processing
    thread = threading.Thread(
        target=process_pipeline_background, 
        args=(pipeline_id, soap_notes, location or 'Not specified')
    )
    thread.start()
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': 'started',
        'message': 'Pipeline processing started'
    })

@app.route('/api/pipeline-status/<pipeline_id>', methods=['GET'])
def get_pipeline_status(pipeline_id):
    """Get the current status of the pipeline"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': pipeline_data['status'],
        'stage': pipeline_data['stage'],
        'error': pipeline_data.get('error'),
        'has_deidentified_text': pipeline_data.get('deidentified_text') is not None,
        'has_cleaned_text': pipeline_data.get('cleaned_text') is not None,
        'has_concepts': pipeline_data.get('concepts_data') is not None
    })

@app.route('/api/get-deidentified-notes/<pipeline_id>', methods=['GET'])
def get_deidentified_notes(pipeline_id):
    """Get de-identified notes if ready"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data['status'] == 'error':
        return jsonify({'error': pipeline_data.get('error', 'Unknown error')}), 500
    
    if pipeline_data.get('deidentified_text') is None:
        return jsonify({'error': 'De-identified notes not ready yet'}), 202
    
    return jsonify({
        'deidentified_text': pipeline_data['deidentified_text'],
        'status': 'success'
    })

@app.route('/api/get-cleaned-notes/<pipeline_id>', methods=['GET'])
def get_cleaned_notes(pipeline_id):
    """Get cleaned notes if ready"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data['status'] == 'error':
        return jsonify({'error': pipeline_data.get('error', 'Unknown error')}), 500
    
    if pipeline_data.get('cleaned_text') is None:
        return jsonify({'error': 'Cleaned notes not ready yet'}), 202
    
    return jsonify({
        'cleaned_text': pipeline_data['cleaned_text'],
        'status': 'success'
    })

@app.route('/api/get-extracted-terms/<pipeline_id>', methods=['GET'])
def get_extracted_terms(pipeline_id):
    """Get extracted terms if ready"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data['status'] == 'error':
        return jsonify({'error': pipeline_data.get('error', 'Unknown error')}), 500
    
    if pipeline_data.get('concepts_data') is None:
        return jsonify({'error': 'Term extraction not ready yet'}), 202
    
    response_data = {
        'concepts_data': pipeline_data['concepts_data'],
        'status': 'success'
    }
    
    return jsonify(response_data)

# LLM-based CFA Endpoints
@app.route('/api/predict-diagnosis/<pipeline_id>', methods=['POST'])
def predict_diagnosis(pipeline_id):
    """Start diagnosis prediction"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('concepts_data') is None:
        return jsonify({'error': 'Clinical terms not available'}), 400
    
    # Check if priority terms are available
    priority_terms = pipeline_data['concepts_data'].get('priority_terms', [])
    if not priority_terms:
        return jsonify({'error': 'No priority terms available for diagnosis prediction'}), 400
    
    # Initialize diagnosis prediction status
    pipeline_storage[pipeline_id]['diagnosis_status'] = 'starting'
    pipeline_storage[pipeline_id]['diagnosis_stage'] = 'Initializing diagnosis prediction...'
    
    # Start background processing
    thread = threading.Thread(target=process_diagnosis_background, args=(pipeline_id,))
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Diagnosis prediction started'
    })

@app.route('/api/diagnosis-status/<pipeline_id>', methods=['GET'])
def get_diagnosis_status(pipeline_id):
    """Get diagnosis prediction status"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': pipeline_data.get('diagnosis_status', 'not_started'),
        'stage': pipeline_data.get('diagnosis_stage', ''),
        'error': pipeline_data.get('diagnosis_error'),
        'has_diagnoses': pipeline_data.get('diagnoses') is not None
    })

@app.route('/api/get-diagnoses/<pipeline_id>', methods=['GET'])
def get_diagnoses(pipeline_id):
    """Get predicted diagnoses"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('diagnosis_status') == 'error':
        return jsonify({'error': pipeline_data.get('diagnosis_error', 'Unknown error')}), 500
    
    if pipeline_data.get('diagnoses') is None:
        return jsonify({'error': 'Diagnoses not ready yet'}), 202
    
    return jsonify({
        'diagnoses': pipeline_data['diagnoses'],
        'status': 'success'
    })

@app.route('/api/predict-cfa/<pipeline_id>', methods=['POST'])
def predict_cfa(pipeline_id):
    """Start CFA prediction for selected diagnoses"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    data = request.get_json()
    selected_diagnoses = data.get('selected_diagnoses', [])
    
    if not selected_diagnoses:
        return jsonify({'error': 'No diagnoses selected'}), 400
    
    # Initialize CFA prediction status
    pipeline_storage[pipeline_id]['cfa_status'] = 'starting'
    pipeline_storage[pipeline_id]['cfa_stage'] = 'Initializing CFA prediction...'
    
    # Start background processing
    thread = threading.Thread(target=process_cfa_background, args=(pipeline_id, selected_diagnoses))
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'CFA prediction started'
    })

@app.route('/api/cfa-status/<pipeline_id>', methods=['GET'])
def get_cfa_status(pipeline_id):
    """Get CFA prediction status"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': pipeline_data.get('cfa_status', 'not_started'),
        'stage': pipeline_data.get('cfa_stage', ''),
        'error': pipeline_data.get('cfa_error'),
        'has_cfa_predictions': pipeline_data.get('cfa_predictions') is not None
    })

@app.route('/api/get-cfa-predictions/<pipeline_id>', methods=['GET'])
def get_cfa_predictions(pipeline_id):
    """Get CFA predictions"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('cfa_status') == 'error':
        return jsonify({'error': pipeline_data.get('cfa_error', 'Unknown error')}), 500
    
    if pipeline_data.get('cfa_predictions') is None:
        return jsonify({'error': 'CFA predictions not ready yet'}), 202
    
    return jsonify({
        'cfa_predictions': pipeline_data['cfa_predictions'],
        'status': 'success'
    })

# Bio-ClinicalBERT Similarity-based CFA Endpoints
@app.route('/api/predict-similarity-cfa/<pipeline_id>', methods=['POST'])
def predict_similarity_cfa(pipeline_id):
    """Start similarity-based CFA prediction using ChromaDB embeddings and Bio-ClinicalBERT"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('concepts_data') is None:
        return jsonify({'error': 'Clinical terms not available'}), 400
    
    # Get priority terms
    priority_terms = pipeline_data['concepts_data'].get('priority_terms', [])
    if not priority_terms:
        return jsonify({'error': 'No priority terms available for similarity-based CFA prediction'}), 400
    
    # Initialize similarity CFA prediction status
    pipeline_storage[pipeline_id]['similarity_cfa_status'] = 'starting'
    pipeline_storage[pipeline_id]['similarity_cfa_stage'] = 'Initializing Bio-ClinicalBERT similarity analysis...'
    
    # Start background processing
    thread = threading.Thread(target=process_similarity_cfa_background, args=(pipeline_id,))
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Similarity-based CFA prediction started',
        'priority_terms_count': len(priority_terms)
    })

@app.route('/api/similarity-cfa-status/<pipeline_id>', methods=['GET'])
def get_similarity_cfa_status(pipeline_id):
    """Get similarity-based CFA prediction status"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': pipeline_data.get('similarity_cfa_status', 'not_started'),
        'stage': pipeline_data.get('similarity_cfa_stage', ''),
        'error': pipeline_data.get('similarity_cfa_error'),
        'has_predictions': pipeline_data.get('similarity_cfa_predictions') is not None
    })

@app.route('/api/get-similarity-cfa-predictions/<pipeline_id>', methods=['GET'])
def get_similarity_cfa_predictions(pipeline_id):
    """Get similarity-based CFA predictions"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('similarity_cfa_status') == 'error':
        return jsonify({'error': pipeline_data.get('similarity_cfa_error', 'Unknown error')}), 500
    
    if pipeline_data.get('similarity_cfa_predictions') is None:
        return jsonify({'error': 'Similarity CFA predictions not ready yet'}), 202
    
    cfa_results = pipeline_data['similarity_cfa_predictions']
    
    return jsonify({
        'cfa_predictions': cfa_results.get('cfa_predictions', []),
        'cfa_percentages': cfa_results.get('cfa_percentages', {}),
        'priority_terms': cfa_results.get('priority_terms', []),
        'total_terms': cfa_results.get('total_terms', 0),
        'message': cfa_results.get('message', ''),
        'status': 'success'
    })

# Visualization and Chart Endpoints
@app.route('/api/get-cfa-visualization/<pipeline_id>', methods=['GET'])
def get_cfa_visualization(pipeline_id):
    """Get 2D visualization of priority terms and CFA terms"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('concepts_data') is None:
        return jsonify({'error': 'Clinical terms not available'}), 400
    
    # Get priority terms
    priority_terms = pipeline_data['concepts_data'].get('priority_terms', [])
    if not priority_terms:
        return jsonify({'error': 'No priority terms available for visualization'}), 400
    
    # Get method from query parameters (default to tsne)
    method = request.args.get('method', 'tsne').lower()
    if method not in ['pca', 'tsne']:
        method = 'tsne'
    
    try:
        # Create visualization using logic.py
        viz_results = create_visualization_data(priority_terms, method)
        
        if "error" in viz_results:
            return jsonify({'error': viz_results["error"]}), 500
        
        return jsonify({
            'visualization': viz_results,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Visualization creation failed: {str(e)}'}), 500

@app.route('/api/get-cfa-prediction-chart/<pipeline_id>', methods=['GET'])
def get_cfa_prediction_chart(pipeline_id):
    """Get chart data for CFA predictions"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('similarity_cfa_predictions') is None:
        return jsonify({'error': 'Similarity CFA predictions not available'}), 400
    
    cfa_results = pipeline_data['similarity_cfa_predictions']
    cfa_predictions = cfa_results.get('cfa_predictions', [])
    
    if not cfa_predictions:
        return jsonify({'error': 'No CFA predictions available for chart'}), 400
    
    # Get top_n from query parameters (default to 10)
    top_n = int(request.args.get('top_n', 10))
    
    try:
        # Create chart data using logic.py
        chart_results = create_prediction_chart_data(cfa_predictions, top_n)
        
        if "error" in chart_results:
            return jsonify({'error': chart_results["error"]}), 500
        
        return jsonify({
            'chart': chart_results,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Chart creation failed: {str(e)}'}), 500

@app.route('/api/get-selectable-cfa-list/<pipeline_id>', methods=['GET'])
def get_selectable_cfa_list_endpoint(pipeline_id):
    """Get CFA predictions as selectable list for UI"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline_data = pipeline_storage[pipeline_id]
    
    if pipeline_data.get('similarity_cfa_predictions') is None:
        return jsonify({'error': 'Similarity CFA predictions not available'}), 400
    
    cfa_results = pipeline_data['similarity_cfa_predictions']
    cfa_predictions = cfa_results.get('cfa_predictions', [])
    
    if not cfa_predictions:
        return jsonify({'error': 'No CFA predictions available'}), 400
    
    try:
        # Create selectable list using logic.py
        selectable_list = get_selectable_cfa_list(cfa_predictions)
        
        return jsonify({
            'selectable_list': selectable_list,
            'total_count': len(selectable_list),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Selectable list creation failed: {str(e)}'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model': CLAUDE_MODEL_ID,
        'medical_nlp_ready': medical_processor is not None,
        'cfa_engine_ready': cfa_engine is not None,
        'components': {
            'aws_bedrock': bedrock_client is not None,
            'medical_nlp': medical_processor is not None,
            'cfa_engine': cfa_engine is not None
        },
        'active_pipelines': len(pipeline_storage)
    })

# Additional utility endpoints for pipeline management
@app.route('/api/pipelines', methods=['GET'])
def list_pipelines():
    """List all active pipelines"""
    try:
        with storage_lock:
            pipeline_list = []
            for pipeline_id, data in pipeline_storage.items():
                pipeline_list.append({
                    'pipeline_id': pipeline_id,
                    'status': data.get('status', 'unknown'),
                    'stage': data.get('stage', ''),
                    'location': data.get('location', ''),
                    'soap_notes_length': len(data.get('soap_notes', '')),
                    'has_deidentified_text': data.get('deidentified_text') is not None,
                    'has_cleaned_text': data.get('cleaned_text') is not None,
                    'has_concepts': data.get('concepts_data') is not None,
                    'has_diagnoses': data.get('diagnoses') is not None,
                    'has_cfa_predictions': data.get('cfa_predictions') is not None,
                    'has_similarity_cfa': data.get('similarity_cfa_predictions') is not None,
                    'diagnosis_status': data.get('diagnosis_status', 'not_started'),
                    'cfa_status': data.get('cfa_status', 'not_started'),
                    'similarity_cfa_status': data.get('similarity_cfa_status', 'not_started'),
                    'errors': {
                        'main_error': data.get('error'),
                        'diagnosis_error': data.get('diagnosis_error'),
                        'cfa_error': data.get('cfa_error'),
                        'similarity_cfa_error': data.get('similarity_cfa_error')
                    },
                    'concept_counts': {
                        'total_concepts': len(data.get('concepts_data', {}).get('all_concepts', [])),
                        'priority_concepts': len(data.get('concepts_data', {}).get('priority_concepts', [])),
                        'priority_terms': len(data.get('concepts_data', {}).get('priority_terms', []))
                    },
                    'prediction_counts': {
                        'diagnoses': len(data.get('diagnoses', [])),
                        'cfa_predictions': len(data.get('cfa_predictions', {})),
                        'similarity_cfa_predictions': len(data.get('similarity_cfa_predictions', {}).get('cfa_predictions', []))
                    }
                })  # <-- This closing bracket was missing
        
        return jsonify({
            'pipelines': pipeline_list,
            'total_count': len(pipeline_list),
            'active_count': len([p for p in pipeline_list if p['status'] in ['starting', 'deidentifying', 'cleaning', 'extracting', 'processing']]),
            'completed_count': len([p for p in pipeline_list if p['status'] == 'completed']),
            'error_count': len([p for p in pipeline_list if p['status'] == 'error']),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Failed to list pipelines: {e}")
        return jsonify({'error': f'Pipeline listing failed: {str(e)}'}), 500

@app.route('/api/cleanup-pipeline/<pipeline_id>', methods=['DELETE'])
def cleanup_pipeline(pipeline_id):
    """Clean up a specific pipeline"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    try:
        with storage_lock:
            del pipeline_storage[pipeline_id]
        
        logger.info(f"Cleaned up pipeline {pipeline_id}")
        return jsonify({
            'message': f'Pipeline {pipeline_id} cleaned up successfully',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Failed to cleanup pipeline: {e}")
        return jsonify({'error': f'Pipeline cleanup failed: {str(e)}'}), 500

@app.route('/api/cleanup-completed-pipelines', methods=['POST'])
def cleanup_completed_pipelines():
    """Clean up all completed pipelines"""
    try:
        cleaned_count = 0
        with storage_lock:
            completed_pipelines = [
                pid for pid, data in pipeline_storage.items() 
                if data.get('status') in ['completed', 'error']
            ]
            
            for pipeline_id in completed_pipelines:
                del pipeline_storage[pipeline_id]
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} completed pipelines")
        return jsonify({
            'message': f'Cleaned up {cleaned_count} completed pipelines',
            'cleaned_count': cleaned_count,
            'remaining_count': len(pipeline_storage),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Failed to cleanup completed pipelines: {e}")
        return jsonify({'error': f'Bulk cleanup failed: {str(e)}'}), 500

@app.route('/api/export-selected-cfas/<pipeline_id>', methods=['POST'])
def export_selected_cfas(pipeline_id):
    """Export selected CFA specialties"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    try:
        data = request.get_json()
        selected_cfas = data.get('selected_cfas', [])
        
        if not selected_cfas:
            return jsonify({'error': 'No CFAs selected for export'}), 400
        
        # Create export data
        export_data = {
            'pipeline_id': pipeline_id,
            'export_timestamp': datetime.now().isoformat(),
            'selected_cfas': selected_cfas,
            'export_type': 'selected_cfa_specialties',
            'total_selected': len(selected_cfas),
            'metadata': {
                'analysis_method': 'bio_clinical_bert_knn_similarity',
                'embedding_model': 'Bio-ClinicalBERT',
                'similarity_method': 'KNN_cosine_similarity'
            }
        }
        
        return jsonify({
            'export_data': export_data,
            'message': f'Successfully exported {len(selected_cfas)} CFA specialties',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Failed to export selected CFAs: {e}")
        return jsonify({'error': f'CFA export failed: {str(e)}'}), 500

@app.route('/api/pipeline-details/<pipeline_id>', methods=['GET'])
def get_pipeline_details(pipeline_id):
    """Get detailed pipeline information"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    try:
        with storage_lock:
            pipeline_data = pipeline_storage[pipeline_id].copy()
        
        # Create detailed summary without large data objects
        summary_data = {
            'pipeline_id': pipeline_id,
            'status': pipeline_data.get('status'),
            'stage': pipeline_data.get('stage'),
            'location': pipeline_data.get('location'),
            'error': pipeline_data.get('error'),
            
            # Status tracking for all stages
            'stages': {
                'main_pipeline': {
                    'status': pipeline_data.get('status', 'not_started'),
                    'stage': pipeline_data.get('stage', '')
                },
                'diagnosis_prediction': {
                    'status': pipeline_data.get('diagnosis_status', 'not_started'),
                    'stage': pipeline_data.get('diagnosis_stage', ''),
                    'error': pipeline_data.get('diagnosis_error')
                },
                'cfa_prediction': {
                    'status': pipeline_data.get('cfa_status', 'not_started'),
                    'stage': pipeline_data.get('cfa_stage', ''),
                    'error': pipeline_data.get('cfa_error')
                },
                'similarity_cfa': {
                    'status': pipeline_data.get('similarity_cfa_status', 'not_started'),
                    'stage': pipeline_data.get('similarity_cfa_stage', ''),
                    'error': pipeline_data.get('similarity_cfa_error')
                }
            },
            
            # Data summaries
            'data_summaries': {
                'original_notes_length': len(pipeline_data.get('soap_notes', '')),
                'deidentified_length': len(pipeline_data.get('deidentified_text', '')),
                'cleaned_length': len(pipeline_data.get('cleaned_text', '')),
                'total_concepts': len(pipeline_data.get('concepts_data', {}).get('all_concepts', [])),
                'priority_concepts': len(pipeline_data.get('concepts_data', {}).get('priority_concepts', [])),
                'priority_terms': len(pipeline_data.get('concepts_data', {}).get('priority_terms', [])),
                'diagnoses_count': len(pipeline_data.get('diagnoses', [])),
                'cfa_predictions_count': len(pipeline_data.get('cfa_predictions', {})),
                'similarity_cfa_count': len(pipeline_data.get('similarity_cfa_predictions', {}).get('cfa_predictions', []))
            },
            
            # Availability flags
            'data_available': {
                'deidentified_text': pipeline_data.get('deidentified_text') is not None,
                'cleaned_text': pipeline_data.get('cleaned_text') is not None,
                'concepts_data': pipeline_data.get('concepts_data') is not None,
                'diagnoses': pipeline_data.get('diagnoses') is not None,
                'cfa_predictions': pipeline_data.get('cfa_predictions') is not None,
                'similarity_cfa_predictions': pipeline_data.get('similarity_cfa_predictions') is not None
            }
        }
        
        return jsonify(summary_data)
        
    except Exception as e:
        logger.error(f"Failed to get pipeline details: {e}")
        return jsonify({'error': f'Pipeline details retrieval failed: {str(e)}'}), 500

@app.route('/api/export-pipeline-results/<pipeline_id>', methods=['GET'])
def export_pipeline_results(pipeline_id):
    """Export comprehensive pipeline results"""
    if pipeline_id not in pipeline_storage:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    try:
        pipeline_data = pipeline_storage[pipeline_id]
        
        # Create comprehensive export
        export_data = {
            'pipeline_metadata': {
                'pipeline_id': pipeline_id,
                'export_timestamp': datetime.now().isoformat(),
                'status': pipeline_data.get('status'),
                'location': pipeline_data.get('location')
            },
            
            'processing_results': {
                'deidentified_text': pipeline_data.get('deidentified_text'),
                'cleaned_text': pipeline_data.get('cleaned_text'),
                'concepts_data': pipeline_data.get('concepts_data'),
                'diagnoses': pipeline_data.get('diagnoses'),
                'cfa_predictions': pipeline_data.get('cfa_predictions'),
                'similarity_cfa_predictions': pipeline_data.get('similarity_cfa_predictions')
            },
            
            'processing_metadata': {
                'main_pipeline_status': pipeline_data.get('status'),
                'diagnosis_status': pipeline_data.get('diagnosis_status'),
                'cfa_status': pipeline_data.get('cfa_status'),
                'similarity_cfa_status': pipeline_data.get('similarity_cfa_status'),
                'errors': {
                    'main_error': pipeline_data.get('error'),
                    'diagnosis_error': pipeline_data.get('diagnosis_error'),
                    'cfa_error': pipeline_data.get('cfa_error'),
                    'similarity_cfa_error': pipeline_data.get('similarity_cfa_error')
                }
            }
        }
        
        return jsonify({
            'export_data': export_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Failed to export pipeline results: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Application startup and main execution
if __name__ == '__main__':
    try:
        # Configuration from environment variables
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print("\n" + "="*80)
        print("🏥 MEDICAL NLP PIPELINE WITH BIO-CLINICALBERT CFA")
        print("="*80)
        print(f"🚀 Starting Flask application on {host}:{port}")
        print(f"🔧 Debug mode: {debug}")
        print(f"🏥 Medical NLP Ready: {medical_processor is not None}")
        print(f"🧠 CFA Engine Ready: {cfa_engine is not None}")
        print(f"☁️  AWS Bedrock Ready: {bedrock_client is not None}")
        print("="*80)
        
        # Display available endpoints
        print("\n📡 Available API Endpoints:")
        print("  Core Pipeline:")
        print("    POST /api/start-pipeline - Start main processing")
        print("    GET  /api/pipeline-status/<id> - Check pipeline status")
        print("    GET  /api/get-deidentified-notes/<id> - Get de-identified text")
        print("    GET  /api/get-cleaned-notes/<id> - Get cleaned text")
        print("    GET  /api/get-extracted-terms/<id> - Get clinical concepts")
        print("\n  LLM-based CFA:")
        print("    POST /api/predict-diagnosis/<id> - Start diagnosis prediction")
        print("    GET  /api/diagnosis-status/<id> - Check diagnosis status")
        print("    GET  /api/get-diagnoses/<id> - Get predicted diagnoses")
        print("    POST /api/predict-cfa/<id> - Start CFA prediction")
        print("    GET  /api/cfa-status/<id> - Check CFA status")
        print("    GET  /api/get-cfa-predictions/<id> - Get CFA predictions")
        print("\n  Bio-ClinicalBERT CFA:")
        print("    POST /api/predict-similarity-cfa/<id> - Start similarity CFA")
        print("    GET  /api/similarity-cfa-status/<id> - Check similarity status")
        print("    GET  /api/get-similarity-cfa-predictions/<id> - Get similarity predictions")
        print("    GET  /api/get-cfa-visualization/<id> - Get PCA/t-SNE visualization")
        print("    GET  /api/get-cfa-prediction-chart/<id> - Get prediction chart")
        print("    GET  /api/get-selectable-cfa-list/<id> - Get selectable CFA list")
        print("\n  Utilities:")
        print("    GET  /api/health - Health check")
        print("    GET  /api/pipelines - List all pipelines")
        print("    GET  /api/pipeline-details/<id> - Get detailed pipeline info")
        print("    DELETE /api/cleanup-pipeline/<id> - Clean up pipeline")
        print("    POST /api/cleanup-completed-pipelines - Bulk cleanup")
        print("    POST /api/export-selected-cfas/<id> - Export selected CFAs")
        print("    GET  /api/export-pipeline-results/<id> - Export all results")
        print("="*80)
        
        # Display component status
        print(f"\n🔍 Component Status:")
        print(f"  ✅ Flask App: Initialized")
        print(f"  {'✅' if bedrock_client else '❌'} AWS Bedrock: {'Ready' if bedrock_client else 'Failed'}")
        print(f"  {'✅' if medical_processor else '❌'} Medical NLP: {'Ready' if medical_processor else 'Failed'}")
        print(f"  {'✅' if cfa_engine else '❌'} CFA Engine: {'Ready' if cfa_engine else 'Failed'}")
        
        if medical_processor is None:
            print("\n⚠️  WARNING: Medical NLP processor failed to initialize!")
            print("   - Clinical concept extraction will not work")
            print("   - Check spaCy and scispacy installation")
        
        if cfa_engine is None:
            print("\n⚠️  WARNING: CFA prediction engine failed to initialize!")
            print("   - Bio-ClinicalBERT CFA predictions will not work")
            print("   - Check ChromaDB and model files")
        
        print("="*80)
        print("🌐 Frontend: Open http://localhost:5000 in your browser")
        print("📚 API Documentation: All endpoints are available at /api/")
        print("🔧 Health Check: GET /api/health")
        print("="*80)
        
        # *** MAIN FLASK APP EXECUTION ***
        app.run(
            host=host,           # Default: 0.0.0.0 (all interfaces)
            port=port,           # Default: 5000
            debug=debug,         # Default: False
            threaded=True,       # Enable multi-threading for concurrent requests
            use_reloader=False   # Disable reloader to prevent double initialization
        )
        
    except KeyboardInterrupt:
        print("\n⛔ Application shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        print("🔚 Application shutdown complete")