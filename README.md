# SOAP Notes Analyzer

## Features

- **🧹 SOAP Notes Cleaning**: Automated cleaning and standardization of SOAP notes using Bedrock model

- **🔬 Clinical Concept Extraction**: Medical NLP using spaCy and scispaCy that extracts keywords from SOAP notes and maps to UMLS terms

- **🩺 Diagnosis Prediction**: Diagnosis suggestions based on extracted clinical concepts using Bedrock model

- **🏥 CFA Prediction**: Medical specialty recommendation that maps diagnoses to a list of predefined medical specialties (CFAs) using Bedrock model

## Installation & Setup

### 1. Clone Repository

### 2. Create Virtual Environment (optional)

### 3. Install Dependencies
```bash
# Install main requirements
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_sci_sm
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

### 4. Set up AWS Credentials

### 5. Start Application
```bash
python app.py
```