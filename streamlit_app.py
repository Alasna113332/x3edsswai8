import streamlit as st
import boto3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import uuid
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
import traceback

# Configure page
st.set_page_config(
    page_title="Medical NLP Pipeline with Bio-ClinicalBERT CFA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your modules (you'll need to ensure these are available)
try:
    from medical_nlp import process_soap_for_cfa, get_medical_processor, extract_priority_terms_list
    from cleaning import clean_soap_notes, validate_cleaned_text
    from cfa import predict_diagnoses, predict_cfa_specialties, validate_diagnoses, validate_cfa_predictions
    from deidentification import deidentify_soap_notes, validate_deidentified_text
    from logic import (
        predict_cfa_from_priority_terms, 
        create_visualization_data, 
        create_prediction_chart_data,
        get_selectable_cfa_list,
        get_cfa_engine,
        convert_numpy_to_python
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Module import error: {e}")
    st.error("Please ensure all required modules are available in your deployment")
    MODULES_AVAILABLE = False

# AWS Configuration
@st.cache_resource
def initialize_aws_session():
    """Initialize AWS session and Bedrock client"""
    try:
        # For Streamlit Cloud deployment, use environment variables or secrets
        # You'll need to configure these in Streamlit Cloud secrets
        if hasattr(st, 'secrets') and 'aws' in st.secrets:
            session = boto3.Session(
                aws_access_key_id=st.secrets['aws']['access_key_id'],
                aws_secret_access_key=st.secrets['aws']['secret_access_key'],
                region_name=st.secrets['aws']['region']
            )
        else:
            # Fallback to default profile (for local development)
            session = boto3.Session(profile_name="PowerUserAccess-148761655243")
        
        bedrock_client = session.client("bedrock-runtime", region_name="ca-central-1")
        return bedrock_client
    except Exception as e:
        st.error(f"❌ AWS initialization failed: {e}")
        return None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize medical processor and CFA engine"""
    components = {
        'bedrock_client': None,
        'medical_processor': None,
        'cfa_engine': None,
        'model_id': "anthropic.claude-3-sonnet-20240229-v1:0",
        'guardrail_id': 'wym574vww1er',
        'guardrail_version': '2'
    }
    
    if not MODULES_AVAILABLE:
        return components
    
    # Initialize AWS
    components['bedrock_client'] = initialize_aws_session()
    
    # Initialize medical processor
    try:
        components['medical_processor'] = get_medical_processor()
        st.success("✅ Medical NLP processor initialized")
    except Exception as e:
        st.warning(f"⚠️ Medical NLP processor failed: {e}")
    
    # Initialize CFA engine
    try:
        components['cfa_engine'] = get_cfa_engine()
        st.success("✅ CFA prediction engine initialized")
    except Exception as e:
        st.warning(f"⚠️ CFA engine failed: {e}")
    
    return components

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'pipeline_data' not in st.session_state:
        st.session_state.pipeline_data = {}
    
    if 'current_pipeline_id' not in st.session_state:
        st.session_state.current_pipeline_id = None
    
    if 'processing_stage' not in st.session_state:
        st.session_state.processing_stage = 'ready'

def create_pipeline_status_display(pipeline_data: Dict):
    """Create a visual status display for the pipeline"""
    stages = [
        ('input', 'Input SOAP Notes', 'ready'),
        ('deidentify', 'De-identification', pipeline_data.get('deidentified_text') is not None),
        ('clean', 'Cleaning', pipeline_data.get('cleaned_text') is not None),
        ('extract', 'Term Extraction', pipeline_data.get('concepts_data') is not None),
        ('diagnose', 'Diagnosis Prediction', pipeline_data.get('diagnoses') is not None),
        ('cfa', 'CFA Prediction', pipeline_data.get('similarity_cfa_predictions') is not None)
    ]
    
    cols = st.columns(len(stages))
    
    for i, (stage_id, stage_name, completed) in enumerate(stages):
        with cols[i]:
            if completed:
                st.success(f"✅ {stage_name}")
            elif pipeline_data.get('status') == 'error':
                st.error(f"❌ {stage_name}")
            else:
                st.info(f"⏳ {stage_name}")

def process_soap_notes(soap_notes: str, location: str, components: Dict) -> str:
    """Process SOAP notes through the pipeline"""
    if not MODULES_AVAILABLE or not components['bedrock_client']:
        st.error("❌ Required components not available")
        return None
    
    pipeline_id = str(uuid.uuid4())
    st.session_state.current_pipeline_id = pipeline_id
    
    # Initialize pipeline data
    pipeline_data = {
        'pipeline_id': pipeline_id,
        'status': 'processing',
        'soap_notes': soap_notes,
        'location': location,
        'created_at': datetime.now().isoformat()
    }
    
    st.session_state.pipeline_data[pipeline_id] = pipeline_data
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: De-identification
        status_text.text("🔒 De-identifying SOAP notes...")
        progress_bar.progress(0.2)
        
        deidentified_text = deidentify_soap_notes(
            bedrock_client=components['bedrock_client'],
            guardrail_id=components['guardrail_id'],
            guardrail_version=components['guardrail_version'],
            soap_notes=soap_notes
        )
        
        validation_result = validate_deidentified_text(deidentified_text)
        if not validation_result['valid']:
            st.error(f"❌ De-identification validation failed: {', '.join(validation_result['issues'])}")
            return None
        
        pipeline_data['deidentified_text'] = deidentified_text
        
        # Step 2: Cleaning
        status_text.text("🧹 Cleaning notes with Claude AI...")
        progress_bar.progress(0.4)
        
        cleaned_text = clean_soap_notes(
            bedrock_client=components['bedrock_client'],
            model_id=components['model_id'],
            soap_notes=deidentified_text,
            location=location
        )
        
        validation_result = validate_cleaned_text(cleaned_text)
        if not validation_result['valid']:
            st.error(f"❌ Cleaning validation failed: {', '.join(validation_result['issues'])}")
            return None
        
        pipeline_data['cleaned_text'] = cleaned_text
        
        # Step 3: Extract clinical concepts
        status_text.text("🧠 Extracting clinical concepts with spaCy...")
        progress_bar.progress(0.6)
        
        if not components['medical_processor']:
            st.error("❌ Medical NLP processor not available")
            return None
        
        cfa_result = process_soap_for_cfa(cleaned_text)
        
        if cfa_result['concepts_dataframe'].empty:
            concepts_data = {
                'all_concepts': [],
                'priority_concepts': [],
                'priority_terms': [],
                'summary': cfa_result['summary']
            }
        else:
            priority_terms = extract_priority_terms_list(cfa_result['priority_concepts_dataframe'])
            concepts_data = {
                'all_concepts': cfa_result['concepts_dataframe'].to_dict('records'),
                'priority_concepts': cfa_result['priority_concepts_dataframe'].to_dict('records'),
                'priority_terms': priority_terms,
                'summary': cfa_result['summary']
            }
        
        pipeline_data['concepts_data'] = concepts_data
        
        # Step 4: Predict diagnoses
        status_text.text("🩺 Predicting diagnoses with Claude AI...")
        progress_bar.progress(0.8)
        
        if concepts_data['priority_terms']:
            diagnoses = predict_diagnoses(
                bedrock_client=components['bedrock_client'],
                model_id=components['model_id'],
                priority_terms=concepts_data['priority_terms']
            )
            
            validation_result = validate_diagnoses(diagnoses)
            if not validation_result['valid']:
                st.warning(f"⚠️ Diagnosis validation issues: {', '.join(validation_result['issues'])}")
            
            pipeline_data['diagnoses'] = diagnoses
        
        # Step 5: Similarity-based CFA prediction
        status_text.text("🎯 Predicting CFA specialties with Bio-ClinicalBERT...")
        progress_bar.progress(0.9)
        
        if concepts_data['priority_terms']:
            cfa_results = predict_cfa_from_priority_terms(concepts_data['priority_terms'])
            
            if "error" not in cfa_results:
                pipeline_data['similarity_cfa_predictions'] = cfa_results
        
        # Complete
        status_text.text("✅ Processing completed!")
        progress_bar.progress(1.0)
        
        pipeline_data['status'] = 'completed'
        st.session_state.pipeline_data[pipeline_id] = pipeline_data
        
        return pipeline_id
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        pipeline_data['status'] = 'error'
        pipeline_data['error'] = str(e)
        st.session_state.pipeline_data[pipeline_id] = pipeline_data
        return None

def display_results_tab(pipeline_data: Dict, components: Dict):
    """Display results in organized tabs"""
    if not pipeline_data or pipeline_data.get('status') != 'completed':
        st.info("⏳ Complete processing to view results")
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Processed Text", 
        "🧠 Clinical Concepts", 
        "🩺 Diagnoses", 
        "🎯 CFA Predictions",
        "📊 Visualizations"
    ])
    
    with tab1:
        st.subheader("Processed Text Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**De-identified Text:**")
            if pipeline_data.get('deidentified_text'):
                st.text_area(
                    "De-identified SOAP Notes",
                    pipeline_data['deidentified_text'],
                    height=200
                )
            else:
                st.info("De-identified text not available")
        
        with col2:
            st.write("**Cleaned Text:**")
            if pipeline_data.get('cleaned_text'):
                st.text_area(
                    "Cleaned SOAP Notes",
                    pipeline_data['cleaned_text'],
                    height=200
                )
            else:
                st.info("Cleaned text not available")
    
    with tab2:
        st.subheader("Clinical Concepts")
        
        concepts_data = pipeline_data.get('concepts_data', {})
        
        if concepts_data.get('priority_terms'):
            st.write(f"**Priority Terms ({len(concepts_data['priority_terms'])}):**")
            
            # Display as tags
            priority_terms = concepts_data['priority_terms']
            term_cols = st.columns(min(4, len(priority_terms)))
            
            for i, term in enumerate(priority_terms):
                with term_cols[i % len(term_cols)]:
                    st.code(term)
            
            # Display detailed concepts
            if concepts_data.get('priority_concepts'):
                st.write("**Detailed Priority Concepts:**")
                priority_df = pd.DataFrame(concepts_data['priority_concepts'])
                st.dataframe(priority_df, use_container_width=True)
            
            if concepts_data.get('all_concepts'):
                with st.expander(f"All Concepts ({len(concepts_data['all_concepts'])})"):
                    all_df = pd.DataFrame(concepts_data['all_concepts'])
                    st.dataframe(all_df, use_container_width=True)
        else:
            st.info("No clinical concepts extracted")
    
    with tab3:
        st.subheader("Predicted Diagnoses")
        
        diagnoses = pipeline_data.get('diagnoses', [])
        
        if diagnoses:
            st.write(f"**{len(diagnoses)} Potential Diagnoses:**")
            
            for i, diagnosis in enumerate(diagnoses, 1):
                with st.container():
                    st.write(f"**{i}. {diagnosis}**")
                    if i < len(diagnoses):
                        st.divider()
        else:
            st.info("No diagnoses predicted")
    
    with tab4:
        st.subheader("CFA Specialty Predictions")
        
        similarity_cfa = pipeline_data.get('similarity_cfa_predictions', {})
        
        if similarity_cfa and similarity_cfa.get('cfa_predictions'):
            cfa_predictions = similarity_cfa['cfa_predictions']
            
            st.write(f"**Top {len(cfa_predictions)} CFA Specialty Predictions:**")
            
            # Create DataFrame for better display
            cfa_df = pd.DataFrame(cfa_predictions)
            
            # Display as a styled table
            st.dataframe(
                cfa_df.style.format({'percentage': '{:.2f}%'}),
                use_container_width=True
            )
            
            # Selection interface
            st.write("**Select CFA Specialties for Export:**")
            
            selected_cfas = []
            for pred in cfa_predictions[:10]:  # Show top 10 for selection
                if st.checkbox(
                    f"{pred['specialty']} ({pred['percentage']:.1f}%)",
                    key=f"cfa_{pred['specialty']}"
                ):
                    selected_cfas.append(pred['specialty'])
            
            if selected_cfas and st.button("📥 Export Selected CFAs"):
                export_data = {
                    'selected_cfas': selected_cfas,
                    'export_timestamp': datetime.now().isoformat(),
                    'pipeline_id': pipeline_data['pipeline_id'],
                    'total_selected': len(selected_cfas)
                }
                
                st.download_button(
                    label="💾 Download Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"cfa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No CFA predictions available")
    
    with tab5:
        st.subheader("Data Visualizations")
        
        concepts_data = pipeline_data.get('concepts_data', {})
        similarity_cfa = pipeline_data.get('similarity_cfa_predictions', {})
        
        if concepts_data.get('priority_terms') and similarity_cfa.get('cfa_predictions'):
            priority_terms = concepts_data['priority_terms']
            cfa_predictions = similarity_cfa['cfa_predictions']
            
            # Visualization method selection
            viz_method = st.selectbox(
                "Select Visualization Method:",
                ["t-SNE", "PCA"],
                key="viz_method"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**CFA Prediction Chart:**")
                try:
                    chart_data = create_prediction_chart_data(cfa_predictions, top_n=10)
                    
                    if "error" not in chart_data and chart_data.get('figure_json'):
                        fig_dict = json.loads(chart_data['figure_json'])
                        fig = go.Figure(fig_dict)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to create prediction chart")
                except Exception as e:
                    st.error(f"Chart creation error: {e}")
            
            with col2:
                st.write(f"**{viz_method} Visualization:**")
                try:
                    viz_data = create_visualization_data(priority_terms, viz_method.lower())
                    
                    if "error" not in viz_data and viz_data.get('figure_json'):
                        fig_dict = json.loads(viz_data['figure_json'])
                        fig = go.Figure(fig_dict)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Failed to create {viz_method} visualization")
                except Exception as e:
                    st.error(f"Visualization error: {e}")
        else:
            st.info("Visualizations will be available after successful CFA prediction")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🏥 Medical NLP Pipeline with Bio-ClinicalBERT CFA")
    st.markdown("---")
    
    # Initialize components
    components = initialize_components()
    
    # Sidebar for navigation and status
    with st.sidebar:
        st.header("🚀 Pipeline Control")
        
        # Component status
        st.subheader("📊 System Status")
        
        aws_status = "✅ Ready" if components['bedrock_client'] else "❌ Failed"
        nlp_status = "✅ Ready" if components['medical_processor'] else "❌ Failed"
        cfa_status = "✅ Ready" if components['cfa_engine'] else "❌ Failed"
        
        st.write(f"**AWS Bedrock:** {aws_status}")
        st.write(f"**Medical NLP:** {nlp_status}")
        st.write(f"**CFA Engine:** {cfa_status}")
        
        st.divider()
        
        # Pipeline management
        st.subheader("📋 Active Pipelines")
        
        if st.session_state.pipeline_data:
            for pipeline_id, data in st.session_state.pipeline_data.items():
                status_emoji = "✅" if data.get('status') == 'completed' else "⏳" if data.get('status') == 'processing' else "❌"
                st.write(f"{status_emoji} {pipeline_id[:8]}...")
        else:
            st.info("No active pipelines")
        
        if st.button("🧹 Clear All Pipelines"):
            st.session_state.pipeline_data = {}
            st.session_state.current_pipeline_id = None
            st.rerun()
    
    # Main content area
    st.header("📝 SOAP Notes Input")
    
    # Input form
    with st.form("soap_input_form"):
        soap_notes = st.text_area(
            "Enter SOAP Notes:",
            height=200,
            placeholder="Enter the medical SOAP notes here..."
        )
        
        location = st.text_input(
            "Location (optional):",
            placeholder="e.g., Emergency Department, ICU, etc."
        )
        
        submitted = st.form_submit_button("🚀 Start Processing")
        
        if submitted:
            if not soap_notes.strip():
                st.error("❌ Please enter SOAP notes")
            elif not MODULES_AVAILABLE:
                st.error("❌ Required modules not available")
            else:
                with st.spinner("🔄 Processing SOAP notes..."):
                    pipeline_id = process_soap_notes(soap_notes, location, components)
                    
                    if pipeline_id:
                        st.success("✅ Processing completed successfully!")
                        st.balloons()
    
    # Display current pipeline status
    if st.session_state.current_pipeline_id and st.session_state.current_pipeline_id in st.session_state.pipeline_data:
        current_data = st.session_state.pipeline_data[st.session_state.current_pipeline_id]
        
        st.header("📊 Pipeline Status")
        create_pipeline_status_display(current_data)
        
        st.header("📋 Results")
        display_results_tab(current_data, components)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🏥 Medical NLP Pipeline with Bio-ClinicalBERT CFA Analysis<br>
            Built with Streamlit • Powered by AWS Bedrock Claude AI
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()