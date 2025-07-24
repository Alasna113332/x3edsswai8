// DOM elements
const soapNotesInput = document.getElementById('soapNotes');
const locationInput = document.getElementById('location');
const startPipelineBtn = document.getElementById('startPipeline');
const pipelineStatus = document.getElementById('pipelineStatus');
const statusText = document.getElementById('statusText');
const progressFill = document.getElementById('progressFill');
const outputSection = document.getElementById('outputSection');

const displayCleanedNotesBtn = document.getElementById('displayCleanedNotes');
const displayExtractedTermsBtn = document.getElementById('displayExtractedTerms');
const cleanedOutput = document.getElementById('cleanedOutput');
const termsOutput = document.getElementById('termsOutput');

const displayDeidentifiedNotesBtn = document.getElementById('displayDeidentifiedNotes');
const deidentifiedOutput = document.getElementById('deidentifiedOutput');

// CFA Method Selection Elements
const cfaMethodSection = document.getElementById('cfaMethodSection');
const llmCfaBtn = document.getElementById('llmCfaBtn');
const bioClinicalCfaBtn = document.getElementById('bioClinicalCfaBtn');

// LLM CFA Elements
const llmCfaSection = document.getElementById('llmCfaSection');
const predictDiagnosisBtn = document.getElementById('predictDiagnosisBtn');
const llmDiagnosisOutput = document.getElementById('llmDiagnosisOutput');
const predictLlmCfaBtn = document.getElementById('predictLlmCfaBtn');
const llmCfaOutput = document.getElementById('llmCfaOutput');

// Bio-ClinicalBERT CFA Elements
const bioClinicalCfaSection = document.getElementById('bioClinicalCfaSection');
const displayVisualizationBtn = document.getElementById('displayVisualizationBtn');
const visualizationOutput = document.getElementById('visualizationOutput');
const predictBioCfaBtn = document.getElementById('predictBioCfaBtn');
const bioCfaOutput = document.getElementById('bioCfaOutput');

// State management
let currentPipelineId = null;
let statusCheckInterval = null;
let diagnosisStatusInterval = null;
let cfaStatusInterval = null;
let bioCfaStatusInterval = null;
let selectedCfaMethod = null;

// API base URL
const API_BASE_URL = 'http://localhost:5000/api';

// Event listeners
startPipelineBtn.addEventListener('click', handleStartPipeline);
displayCleanedNotesBtn.addEventListener('click', handleDisplayCleanedNotes);
displayExtractedTermsBtn.addEventListener('click', handleDisplayExtractedTerms);
displayDeidentifiedNotesBtn.addEventListener('click', handleDisplayDeidentifiedNotes);

// CFA Method Selection
llmCfaBtn.addEventListener('click', () => selectCfaMethod('llm'));
bioClinicalCfaBtn.addEventListener('click', () => selectCfaMethod('bioclinical'));

// LLM CFA Event Listeners
predictDiagnosisBtn.addEventListener('click', handlePredictDiagnosis);
predictLlmCfaBtn.addEventListener('click', handlePredictLlmCfa);

// Bio-ClinicalBERT CFA Event Listeners
displayVisualizationBtn.addEventListener('click', handleDisplayVisualization);
predictBioCfaBtn.addEventListener('click', handlePredictBioClinicalCfa);

// CFA Method Selection Function
function selectCfaMethod(method) {
    selectedCfaMethod = method;
    
    // Update button states
    if (method === 'llm') {
        llmCfaBtn.classList.add('active');
        bioClinicalCfaBtn.classList.remove('active');
        llmCfaSection.classList.remove('hidden');
        bioClinicalCfaSection.classList.add('hidden');
        
        // Enable diagnosis prediction for LLM method
        predictDiagnosisBtn.disabled = false;
    } else {
        bioClinicalCfaBtn.classList.add('active');
        llmCfaBtn.classList.remove('active');
        bioClinicalCfaSection.classList.remove('hidden');
        llmCfaSection.classList.add('hidden');
        
        // Enable visualization for Bio-ClinicalBERT method
        displayVisualizationBtn.disabled = false;
    }
}

// Main Pipeline Function
async function handleStartPipeline() {
    const soapNotes = soapNotesInput.value.trim();
    const location = locationInput.value.trim();

    if (!soapNotes) {
        alert('Please enter SOAP notes.');
        return;
    }

    // Reset UI
    resetUI();
    
    // Show pipeline status and output section
    pipelineStatus.classList.remove('hidden');
    outputSection.classList.remove('hidden');
    
    // Disable start button
    startPipelineBtn.disabled = true;
    startPipelineBtn.textContent = 'Processing...';

    try {
        // Start the pipeline
        const response = await startPipeline(soapNotes, location);
        currentPipelineId = response.pipeline_id;
        
        // Start monitoring pipeline status
        startStatusMonitoring();
        
    } catch (error) {
        showError('Failed to start pipeline: ' + error.message);
        resetButtons();
    }
}

function startStatusMonitoring() {
    statusCheckInterval = setInterval(async () => {
        try {
            const status = await getPipelineStatus(currentPipelineId);
            updateStatusDisplay(status);
            
            // Update display button states based on what's ready
            if (status.has_deidentified_text) {
                displayDeidentifiedNotesBtn.classList.remove('hidden');
            }
            
            if (status.has_cleaned_text) {
                displayCleanedNotesBtn.classList.remove('hidden');
            }
            
            if (status.has_concepts) {
                displayExtractedTermsBtn.classList.remove('hidden');
                // Show CFA method selection after terms are extracted
                cfaMethodSection.classList.remove('hidden');
            }
            
            // Stop monitoring if completed or errored
            if (status.status === 'completed' || status.status === 'error') {
                clearInterval(statusCheckInterval);
                resetButtons();
                
                if (status.status === 'error') {
                    showError('Pipeline error: ' + status.error);
                }
            }
            
        } catch (error) {
            console.error('Status check failed:', error);
            clearInterval(statusCheckInterval);
            resetButtons();
        }
    }, 2000); // Check every 2 seconds
}

function updateStatusDisplay(status) {
    statusText.textContent = status.stage || 'Processing...';
    
    // Update progress bar based on status
    let progress = 0;
    switch (status.status) {
        case 'starting':
            progress = 5;
            break;
        case 'deidentifying':
            progress = 25;
            break;
        case 'cleaning':
            progress = 50;
            break;
        case 'extracting':
            progress = 85;
            break;
        case 'completed':
            progress = 100;
            setTimeout(() => {
                pipelineStatus.classList.add('hidden');
            }, 2000);
            break;
        case 'error':
            progress = 100;
            statusText.textContent = 'Error: ' + (status.error || 'Unknown error');
            progressFill.style.backgroundColor = '#dc3545';
            break;
    }
    
    progressFill.style.width = progress + '%';
}

// Display Functions
async function handleDisplayDeidentifiedNotes() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    deidentifiedOutput.classList.remove('hidden');
    deidentifiedOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Loading de-identified notes...</span>
        </div>
    `;

    try {
        const response = await getDeidentifiedNotes(currentPipelineId);
        
        if (response.deidentified_text) {
            deidentifiedOutput.innerHTML = `<div class="content-text">${response.deidentified_text}</div>`;
        } else {
            throw new Error('No de-identified text available');
        }
        
    } catch (error) {
        if (error.message.includes('not ready')) {
            deidentifiedOutput.innerHTML = `
                <div class="not-ready-message">
                    <span>De-identified notes are still being processed. Please wait...</span>
                </div>
            `;
        } else {
            deidentifiedOutput.innerHTML = `
                <div class="error-message">
                    <span>Error loading de-identified notes: ${error.message}</span>
                </div>
            `;
        }
    }
}

async function handleDisplayCleanedNotes() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    cleanedOutput.classList.remove('hidden');
    cleanedOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Loading cleaned notes...</span>
        </div>
    `;

    try {
        const response = await getCleanedNotes(currentPipelineId);
        
        if (response.cleaned_text) {
            cleanedOutput.innerHTML = `<div class="content-text">${response.cleaned_text}</div>`;
        } else {
            throw new Error('No cleaned text available');
        }
        
    } catch (error) {
        if (error.message.includes('not ready')) {
            cleanedOutput.innerHTML = `
                <div class="not-ready-message">
                    <span>Cleaned notes are still being processed. Please wait...</span>
                </div>
            `;
        } else {
            cleanedOutput.innerHTML = `
                <div class="error-message">
                    <span>Error loading cleaned notes: ${error.message}</span>
                </div>
            `;
        }
    }
}

async function handleDisplayExtractedTerms() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    termsOutput.classList.remove('hidden');
    termsOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Loading extracted terms...</span>
        </div>
    `;

    try {
        const response = await getExtractedTerms(currentPipelineId);
        
        if (response.concepts_data) {
            displayTermsResults(response.concepts_data);
        } else {
            throw new Error('No extracted terms available');
        }
        
    } catch (error) {
        if (error.message.includes('not ready')) {
            termsOutput.innerHTML = `
                <div class="not-ready-message">
                    <span>Term extraction is still in progress. Please wait...</span>
                </div>
            `;
        } else {
            termsOutput.innerHTML = `
                <div class="error-message">
                    <span>Error loading extracted terms: ${error.message}</span>
                </div>
            `;
        }
    }
}

function displayTermsResults(conceptsData) {
    const concepts = conceptsData.all_concepts;
    const summary = conceptsData.summary;

    if (!concepts || concepts.length === 0) {
        termsOutput.innerHTML = `
            <div class="no-concepts">
                <h4>No clinical concepts found</h4>
                <p>Try entering more detailed medical notes with symptoms, diagnoses, or treatments.</p>
            </div>
        `;
        return;
    }

    let html = `
        <div class="summary">
            <h4>Analysis Summary</h4>
            <div class="summary-grid">
                <div class="summary-item">
                    <strong>${summary.total_concepts}</strong>
                    <span>Total Concepts</span>
                </div>
                <div class="summary-item">
                    <strong>${summary.priority_concepts}</strong>
                    <span>Priority Concepts</span>
                </div>
                <div class="summary-item">
                    <strong>${summary.score_range.min.toFixed(3)} - ${summary.score_range.max.toFixed(3)}</strong>
                    <span>Score Range</span>
                </div>
            </div>
        </div>

        <h4 style="color: #1e40af; margin: 20px 0 15px 0;">Extracted Clinical Concepts DataFrame</h4>
        <div class="dataframe-container">
            <table class="dataframe-table">
                <thead>
                    <tr>
                        <th class="index-header"></th>
                        <th>Term</th>
                        <th>UMLS_CUI</th>
                        <th>Canonical_Name</th>
                        <th>Score</th>
                        <th>Semantic_Types</th>
                    </tr>
                </thead>
                <tbody>
    `;

    concepts.forEach((concept, index) => {
        const priorityClass = concept.Priority ? 'priority-row' : '';

        html += `
            <tr class="${priorityClass}">
                <td class="index-cell">${index}</td>
                <td class="term-cell">${concept.Term}</td>
                <td class="cui-cell">${concept.UMLS_CUI}</td>
                <td class="canonical-cell">${concept.Canonical_Name}</td>
                <td class="score-cell">${concept.Score}</td>
                <td class="semantic-cell">${concept.Semantic_Types}</td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
    `;
    
    termsOutput.innerHTML = html;
}

// LLM CFA Functions
async function handlePredictDiagnosis() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    llmDiagnosisOutput.classList.remove('hidden');
    llmDiagnosisOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Starting LLM diagnosis prediction...</span>
        </div>
    `;

    try {
        await startDiagnosisPrediction(currentPipelineId);
        startDiagnosisStatusMonitoring();
        
    } catch (error) {
        llmDiagnosisOutput.innerHTML = `
            <div class="error-message">
                <span>Error starting diagnosis prediction: ${error.message}</span>
            </div>
        `;
    }
}

function startDiagnosisStatusMonitoring() {
    diagnosisStatusInterval = setInterval(async () => {
        try {
            const status = await getDiagnosisStatus(currentPipelineId);
            
            if (status.status === 'processing') {
                llmDiagnosisOutput.innerHTML = `
                    <div class="loading-animation">
                        <div class="loading-spinner"></div>
                        <span>${status.stage}</span>
                    </div>
                `;
            } else if (status.status === 'completed' && status.has_diagnoses) {
                const diagnosesData = await getDiagnoses(currentPipelineId);
                displayDiagnosisResults(diagnosesData.diagnoses);
                clearInterval(diagnosisStatusInterval);
                
                // Enable LLM CFA prediction button
                predictLlmCfaBtn.disabled = false;
            } else if (status.status === 'error') {
                llmDiagnosisOutput.innerHTML = `
                    <div class="error-message">
                        <span>Diagnosis prediction error: ${status.error}</span>
                    </div>
                `;
                clearInterval(diagnosisStatusInterval);
            }
            
        } catch (error) {
            console.error('Diagnosis status check failed:', error);
            clearInterval(diagnosisStatusInterval);
        }
    }, 2000);
}

function displayDiagnosisResults(diagnoses) {
    let html = `
        <div class="diagnosis-container">
            <h4 style="color: #1e40af; margin-bottom: 15px;">Predicted Diagnoses (LLM)</h4>
            <div class="diagnosis-list">
    `;

    diagnoses.forEach((diagnosis, index) => {
        html += `
            <div class="diagnosis-item">
                <label class="checkbox-label">
                    <input type="checkbox" class="diagnosis-checkbox" value="${diagnosis}" id="diagnosis-${index}">
                    <span class="diagnosis-text">${diagnosis}</span>
                </label>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;
    
    llmDiagnosisOutput.innerHTML = html;
}

async function handlePredictLlmCfa() {
    const checkboxes = document.querySelectorAll('.diagnosis-checkbox:checked');
    const selectedDiagnoses = Array.from(checkboxes).map(cb => cb.value);
    
    if (selectedDiagnoses.length === 0) {
        alert('Please select at least one diagnosis.');
        return;
    }

    llmCfaOutput.classList.remove('hidden');
    llmCfaOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Starting LLM CFA prediction...</span>
        </div>
    `;

    try {
        await startCfaPrediction(currentPipelineId, selectedDiagnoses);
        startCfaStatusMonitoring();
        
    } catch (error) {
        llmCfaOutput.innerHTML = `
            <div class="error-message">
                <span>Error starting CFA prediction: ${error.message}</span>
            </div>
        `;
    }
}

function startCfaStatusMonitoring() {
    cfaStatusInterval = setInterval(async () => {
        try {
            const status = await getCfaStatus(currentPipelineId);
            
            if (status.status === 'processing') {
                llmCfaOutput.innerHTML = `
                    <div class="loading-animation">
                        <div class="loading-spinner"></div>
                        <span>${status.stage}</span>
                    </div>
                `;
            } else if (status.status === 'completed' && status.has_cfa_predictions) {
                const cfaData = await getCfaPredictions(currentPipelineId);
                displayLlmCfaResults(cfaData.cfa_predictions);
                clearInterval(cfaStatusInterval);
            } else if (status.status === 'error') {
                llmCfaOutput.innerHTML = `
                    <div class="error-message">
                        <span>CFA prediction error: ${status.error}</span>
                    </div>
                `;
                clearInterval(cfaStatusInterval);
            }
            
        } catch (error) {
            console.error('CFA status check failed:', error);
            clearInterval(cfaStatusInterval);
        }
    }, 2000);
}

function displayLlmCfaResults(cfaPredictions) {
    let html = `
        <h4 style="color: #1e40af; margin: 20px 0 15px 0;">Medical Specialty Predictions (LLM)</h4>
        <div class="cfa-predictions">
    `;

    for (const [diagnosis, specialty] of Object.entries(cfaPredictions)) {
        html += `
            <div class="cfa-prediction-item">
                <span class="diagnosis-name">${diagnosis}</span>
                <span class="specialty-arrow">→</span>
                <span class="specialty-name">${specialty}</span>
            </div>
        `;
    }

    html += `</div>`;
    llmCfaOutput.innerHTML = html;
}

// Bio-ClinicalBERT CFA Functions
async function handleDisplayVisualization() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    visualizationOutput.classList.remove('hidden');
    visualizationOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Creating 2D visualization with t-SNE...</span>
        </div>
    `;

    try {
        const response = await getCfaVisualization(currentPipelineId, 'tsne');
        
        // DEBUG: Log the full response
        console.log('🔍 Full visualization response:', response);
        console.log('🔍 Visualization data keys:', Object.keys(response.visualization || {}));
        
        if (response.visualization && !response.visualization.error) {
            displayVisualizationResults(response.visualization);
            // Enable Bio-ClinicalBERT CFA prediction button
            predictBioCfaBtn.disabled = false;
        } else {
            throw new Error(response.visualization?.error || 'Unknown visualization error');
        }
        
    } catch (error) {
        console.error('❌ Visualization error:', error);
        visualizationOutput.innerHTML = `
            <div class="error-message">
                <span>Error creating visualization: ${error.message}</span>
                <button onclick="handleDisplayVisualization()" style="margin-left: 10px;">Retry</button>
            </div>
        `;
    }
}

function displayVisualizationResults(vizData) {
    console.log('🎨 [DEBUG] Starting displayVisualizationResults');
    console.log('🎨 [DEBUG] vizData received:', vizData);
    
    if (vizData.error) {
        console.error('❌ [DEBUG] vizData has error:', vizData.error);
        visualizationOutput.innerHTML = `
            <div class="error-message">
                <span>Visualization error: ${vizData.error}</span>
            </div>
        `;
        return;
    }

    const figureJson = vizData.figure_json;
    const method = vizData.method || 'tsne';
    const nPriority = vizData.n_priority || 0;
    const nCfa = vizData.n_cfa || 0;

    console.log('🔍 [DEBUG] Figure JSON exists:', !!figureJson);
    console.log('🔍 [DEBUG] Figure JSON type:', typeof figureJson);
    console.log('🔍 [DEBUG] Figure JSON length:', figureJson ? figureJson.length : 'N/A');
    console.log('🔍 [DEBUG] Method:', method);
    console.log('🔍 [DEBUG] Priority terms:', nPriority);
    console.log('🔍 [DEBUG] CFA terms:', nCfa);

    // Create the HTML structure
    let html = `
        <div class="visualization-container">
            <h4 style="color: #1e40af; margin-bottom: 15px;">
                2D Visualization: Priority Terms vs CFA Terms
            </h4>
            
            <div class="viz-summary">
                <div class="viz-stats">
                    <div class="viz-stat-item">
                        <strong>${nPriority}</strong>
                        <span>Priority Terms</span>
                    </div>
                    <div class="viz-stat-item">
                        <strong>${nCfa}</strong>
                        <span>CFA Terms</span>
                    </div>
                    <div class="viz-stat-item">
                        <strong>${method.toUpperCase()}</strong>
                        <span>Method</span>
                    </div>
                </div>
            </div>
            
            <div class="viz-controls">
                <button id="switchToPcaBtn" class="viz-method-btn ${method === 'pca' ? 'active' : ''}" 
                        data-method="pca">PCA</button>
                <button id="switchToTsneBtn" class="viz-method-btn ${method === 'tsne' ? 'active' : ''}" 
                        data-method="tsne">t-SNE</button>
            </div>
            
            <div id="visualizationPlot" class="visualization-plot" style="width: 100%; height: 600px; border: 2px solid red; background: #f0f0f0;">
                <div style="padding: 20px; text-align: center; color: #666;">
                    🔍 DEBUG: Chart container created - waiting for Plotly...
                </div>
            </div>
            
            <div class="viz-legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF6B6B; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></div>
                    <span>Priority Terms (from medical notes)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ECDC4; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></div>
                    <span>CFA Terms (medical specialties)</span>
                </div>
            </div>
        </div>
    `;
    
    console.log('🔍 [DEBUG] Setting visualizationOutput.innerHTML');
    visualizationOutput.innerHTML = html;
    
    // Check if the element was created
    const plotElement = document.getElementById('visualizationPlot');
    console.log('🔍 [DEBUG] Plot element found:', !!plotElement);
    console.log('🔍 [DEBUG] Plot element:', plotElement);
    
    if (!plotElement) {
        console.error('❌ [DEBUG] visualizationPlot element not found after setting innerHTML!');
        return;
    }
    
    // Check Plotly availability
    console.log('🔍 [DEBUG] Plotly available:', typeof Plotly !== 'undefined');
    console.log('🔍 [DEBUG] Plotly object:', typeof Plotly !== 'undefined' ? Plotly : 'NOT AVAILABLE');
    
    // Function to convert coordinate objects to arrays
    function convertCoordinateObjectToArray(coordObj) {
        console.log('🔧 [CONVERT] Input object:', coordObj);
        console.log('🔧 [CONVERT] Input type:', typeof coordObj);
        console.log('🔧 [CONVERT] Is array:', Array.isArray(coordObj));
        
        if (Array.isArray(coordObj)) {
            console.log('✅ [CONVERT] Already an array, length:', coordObj.length);
            return coordObj; // Already an array
        }
        
        if (coordObj && typeof coordObj === 'object') {
            const keys = Object.keys(coordObj);
            console.log('🔧 [CONVERT] Object keys:', keys);
            
            // Handle NumPy array format with 'bdata' property
            if (keys.includes('bdata') && coordObj.bdata) {
                console.log('🔧 [CONVERT] Found NumPy bdata format');
                console.log('🔧 [CONVERT] bdata type:', typeof coordObj.bdata);
                console.log('🔧 [CONVERT] bdata value:', coordObj.bdata);
                
                if (Array.isArray(coordObj.bdata)) {
                    console.log('✅ [CONVERT] Using bdata array, length:', coordObj.bdata.length);
                    return coordObj.bdata;
                }
            }
            
            // Handle direct numeric object format {0: 1.2, 1: 3.4, 2: 5.6}
            const numericKeys = keys.filter(k => !isNaN(parseInt(k)));
            console.log('🔧 [CONVERT] Numeric keys found:', numericKeys);
            
            if (numericKeys.length > 0) {
                const sortedKeys = numericKeys.map(k => parseInt(k)).sort((a, b) => a - b);
                const arrayValues = sortedKeys.map(k => coordObj[k.toString()]);
                console.log('✅ [CONVERT] Converted from numeric keys, length:', arrayValues.length);
                console.log('🔧 [CONVERT] Sample values:', arrayValues.slice(0, 3));
                return arrayValues;
            }
            
            // Handle case where the object itself might be the array data
            const values = Object.values(coordObj);
            if (values.length > 0 && values.every(v => typeof v === 'number')) {
                console.log('✅ [CONVERT] Using object values as array, length:', values.length);
                return values;
            }
            
            console.log('❌ [CONVERT] No convertible format found');
            console.log('🔧 [CONVERT] All values:', values);
        }
        
        console.log('❌ [CONVERT] Returning empty array as fallback');
        return []; // Fallback to empty array
    }
    
    // Try to render Plotly chart
    if (figureJson) {
        try {
            console.log('🔍 [DEBUG] Attempting to parse figure JSON...');
            console.log('🔍 [DEBUG] Raw JSON preview:', figureJson.substring(0, 200) + '...');
            
            const figure = JSON.parse(figureJson);
            console.log('🔍 [DEBUG] JSON parsed successfully');
            console.log('🔍 [DEBUG] Figure object keys:', Object.keys(figure));
            console.log('🔍 [DEBUG] Figure.data type:', typeof figure.data);
            console.log('🔍 [DEBUG] Figure.data:', figure.data);
            
            // Safe check for data array
            if (figure.data && Array.isArray(figure.data)) {
                console.log('🔍 [DEBUG] Figure.data length:', figure.data.length);
                console.log('🔍 [DEBUG] Figure.layout:', figure.layout);
                
                // Check if we have data points
                if (figure.data.length > 0) {
                    console.log('✅ [DEBUG] Data series found:', figure.data.length);
                    
                    // Examine coordinate objects and fix them
                    console.log('🔍 [DEBUG] === FIXING COORDINATE DATA ===');
                    const fixedFigure = {
                        data: figure.data.map((series, i) => {
                            console.log(`🔍 [DEBUG] Fixing series ${i}...`);
                            console.log(`🔍 [DEBUG] Series ${i} x type:`, typeof series.x);
                            console.log(`🔍 [DEBUG] Series ${i} y type:`, typeof series.y);
                            console.log(`🔍 [DEBUG] Series ${i} x is array:`, Array.isArray(series.x));
                            console.log(`🔍 [DEBUG] Series ${i} y is array:`, Array.isArray(series.y));
                            
                            // Convert coordinates to arrays
                            const fixedX = convertCoordinateObjectToArray(series.x);
                            const fixedY = convertCoordinateObjectToArray(series.y);
                            
                            console.log(`🔍 [DEBUG] Series ${i} fixed x length:`, fixedX.length);
                            console.log(`🔍 [DEBUG] Series ${i} fixed y length:`, fixedY.length);
                            
                            if (fixedX.length > 0 && fixedY.length > 0) {
                                console.log(`🔍 [DEBUG] Series ${i} sample coordinates:`, {
                                    x: fixedX.slice(0, 3),
                                    y: fixedY.slice(0, 3)
                                });
                            }
                            
                            // Return fixed series
                            return {
                                ...series,
                                x: fixedX,
                                y: fixedY,
                                marker: {
                                    ...series.marker,
                                    size: 10, // Ensure visible markers
                                    line: { width: 1, color: 'white' }
                                }
                            };
                        }),
                        layout: {
                            ...figure.layout,
                            margin: { l: 60, r: 60, t: 80, b: 60 },
                            xaxis: {
                                ...figure.layout.xaxis,
                                autorange: true
                            },
                            yaxis: {
                                ...figure.layout.yaxis,
                                autorange: true
                            }
                        }
                    };
                    
                    console.log('🔍 [DEBUG] === END COORDINATE FIXING ===');
                    console.log('🔍 [DEBUG] About to call Plotly.newPlot...');
                    console.log('🔍 [DEBUG] Target element ID: visualizationPlot');
                    
                    // Add a small delay to ensure DOM is ready
                    setTimeout(() => {
                        try {
                            console.log('🔍 [DEBUG] Calling Plotly.newPlot NOW');
                            
                            Plotly.newPlot('visualizationPlot', fixedFigure.data, fixedFigure.layout, {
                                responsive: true,
                                displayModeBar: true,
                                modeBarButtonsToRemove: ['select2d', 'lasso2d'],
                                displaylogo: false
                            }).then(() => {
                                console.log('✅ [DEBUG] Plotly.newPlot SUCCESS!');
                                console.log('✅ [DEBUG] Chart should now be visible with FIXED coordinate data');
                                
                                // Remove debug border
                                const plotEl = document.getElementById('visualizationPlot');
                                if (plotEl) {
                                    plotEl.style.border = 'none';
                                    plotEl.style.background = 'transparent';
                                }
                            }).catch(plotError => {
                                console.error('❌ [DEBUG] Plotly.newPlot FAILED:', plotError);
                                console.error('❌ [DEBUG] Error details:', plotError.message, plotError.stack);
                                document.getElementById('visualizationPlot').innerHTML = `
                                    <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                                        <strong>Plotly Rendering Failed:</strong><br>
                                        ${plotError.message}<br>
                                        <small>Check browser console for details</small>
                                    </div>
                                `;
                            });
                        } catch (syncError) {
                            console.error('❌ [DEBUG] Synchronous error in Plotly call:', syncError);
                            document.getElementById('visualizationPlot').innerHTML = `
                                <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                                    <strong>Sync Error:</strong><br>
                                    ${syncError.message}
                                </div>
                            `;
                        }
                    }, 100); // 100ms delay
                    
                } else {
                    console.error('❌ [DEBUG] No data series in figure.data');
                    console.log('🔍 [DEBUG] figure.data contents:', figure.data);
                    document.getElementById('visualizationPlot').innerHTML = `
                        <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                            <strong>No data series available</strong><br>
                            <small>figure.data is empty array</small>
                        </div>
                    `;
                }
            } else {
                console.error('❌ [DEBUG] figure.data is not an array');
                console.log('🔍 [DEBUG] figure.data type:', typeof figure.data);
                console.log('🔍 [DEBUG] figure.data value:', figure.data);
                document.getElementById('visualizationPlot').innerHTML = `
                    <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                        <strong>Invalid data structure</strong><br>
                        <small>figure.data is not an array</small>
                    </div>
                `;
            }
            
        } catch (parseError) {
            console.error('❌ [DEBUG] JSON parse error:', parseError);
            console.log('🔍 [DEBUG] Error name:', parseError.name);
            console.log('🔍 [DEBUG] Error message:', parseError.message);
            
            document.getElementById('visualizationPlot').innerHTML = `
                <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                    <strong>JSON Parse Error:</strong><br>
                    ${parseError.message}<br>
                    <small>Check console for raw JSON</small>
                </div>
            `;
        }
    } else {
        console.error('❌ [DEBUG] No figure JSON provided');
        console.log('🔍 [DEBUG] vizData keys:', Object.keys(vizData));
        document.getElementById('visualizationPlot').innerHTML = `
            <div class="error-message" style="padding: 20px; color: red; text-align: center;">
                <strong>No chart data received from server</strong><br>
                <small>figure_json is missing or null</small>
            </div>
        `;
    }
    
    // Add event listeners for method switching
    console.log('🔍 [DEBUG] Adding method switch event listeners');
    document.getElementById('switchToPcaBtn')?.addEventListener('click', () => switchVisualizationMethod('pca'));
    document.getElementById('switchToTsneBtn')?.addEventListener('click', () => switchVisualizationMethod('tsne'));
    
    console.log('🔍 [DEBUG] displayVisualizationResults completed');
}

async function switchVisualizationMethod(method) {
    console.log(`🔄 Switching to ${method.toUpperCase()} visualization`);
    
    // Update button states
    document.querySelectorAll('.viz-method-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`switchTo${method.charAt(0).toUpperCase() + method.slice(1)}Btn`).classList.add('active');
    
    // Show loading in plot area
    document.getElementById('visualizationPlot').innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Recalculating with ${method.toUpperCase()}...</span>
        </div>
    `;
    
    try {
        const response = await getCfaVisualization(currentPipelineId, method);
        console.log(`🔍 ${method.toUpperCase()} response:`, response);
        
        // Update only the plot, keep the rest of the UI
        const figureJson = response.visualization.figure_json;
        if (figureJson) {
            const figure = JSON.parse(figureJson);
            console.log(`✅ Rendering ${method.toUpperCase()} chart`);
            
            Plotly.newPlot('visualizationPlot', figure.data, figure.layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['select2d', 'lasso2d'],
                displaylogo: false
            }).then(() => {
                console.log(`✅ ${method.toUpperCase()} chart rendered successfully!`);
            });
        }
        
    } catch (error) {
        console.error(`❌ Error switching to ${method}:`, error);
        document.getElementById('visualizationPlot').innerHTML = `
            <div class="error-message">
                <span>Error switching method: ${error.message}</span>
            </div>
        `;
    }
}

async function handlePredictBioClinicalCfa() {
    if (!currentPipelineId) {
        alert('No active pipeline. Please start processing first.');
        return;
    }

    bioCfaOutput.classList.remove('hidden');
    bioCfaOutput.innerHTML = `
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <span>Initializing Bio-ClinicalBERT similarity analysis...</span>
        </div>
    `;

    try {
        const response = await predictSimilarityCfa(currentPipelineId);
        
        if (response.status === 'started') {
            startBioCfaStatusMonitoring();
        } else {
            throw new Error('Failed to start Bio-ClinicalBERT CFA prediction');
        }
        
    } catch (error) {
        bioCfaOutput.innerHTML = `
            <div class="error-message">
                <span>Error starting Bio-ClinicalBERT CFA: ${error.message}</span>
            </div>
        `;
    }
}

function startBioCfaStatusMonitoring() {
    bioCfaStatusInterval = setInterval(async () => {
        try {
            const status = await getSimilarityCfaStatus(currentPipelineId);
            
            if (status.status === 'processing' || status.status === 'starting') {
                bioCfaOutput.innerHTML = `
                    <div class="loading-animation">
                        <div class="loading-spinner"></div>
                        <span>${status.stage}</span>
                    </div>
                `;
            } else if (status.status === 'completed' && status.has_predictions) {
                const predictionsData = await getSimilarityCfaPredictions(currentPipelineId);
                displayBioCfaResults(predictionsData);
                clearInterval(bioCfaStatusInterval);
            } else if (status.status === 'error') {
                bioCfaOutput.innerHTML = `
                    <div class="error-message">
                        <span>Bio-ClinicalBERT CFA prediction error: ${status.error}</span>
                    </div>
                `;
                clearInterval(bioCfaStatusInterval);
            }
            
        } catch (error) {
            console.error('Bio-ClinicalBERT CFA status check failed:', error);
            clearInterval(bioCfaStatusInterval);
        }
    }, 2000);
}

function displayBioCfaResults(predictionsData) {
    const cfaPredictions = predictionsData.cfa_predictions || [];
    const cfaPercentages = predictionsData.cfa_percentages || {};
    const totalTerms = predictionsData.total_terms || 0;

    if (cfaPredictions.length === 0) {
        bioCfaOutput.innerHTML = `
            <div class="no-predictions">
                <h4>No CFA predictions generated</h4>
                <p>Try processing notes with more medical terminology.</p>
            </div>
        `;
        return;
    }

    // Create header with summary
    let html = `
        <div class="bio-cfa-container">
            <h4 style="color: #1e40af; margin-bottom: 15px;">
                Bio-ClinicalBERT Similarity-Based CFA Predictions 
                <span style="color: #64748b; font-size: 0.9rem; font-weight: normal;">
                    (Based on ${totalTerms} priority terms)
                </span>
            </h4>
            
            <div class="prediction-summary">
                <div class="summary-stats">
                    <div class="stat-item">
                        <strong>${cfaPredictions.length}</strong>
                        <span>Specialties Matched</span>
                    </div>
                    <div class="stat-item">
                        <strong>${totalTerms}</strong>
                        <span>Priority Terms</span>
                    </div>
                    <div class="stat-item">
                        <strong>${cfaPredictions[0]?.percentage || 0}%</strong>
                        <span>Top Match</span>
                    </div>
                </div>
            </div>
    `;

    // Create percentage chart (top 10)
    const top10 = cfaPredictions.slice(0, 10);
    html += `
        <div class="percentage-chart">
            <h5 style="color: #059669; margin-bottom: 15px;">Top 10 CFA Predictions</h5>
            <div class="chart-bars">
    `;

    top10.forEach((prediction, index) => {
        const barWidth = prediction.percentage;
        const rank = index + 1;
        
        html += `
            <div class="chart-bar-item">
                <div class="bar-label">
                    <span class="rank">${rank}.</span>
                    <span class="specialty-name">${prediction.specialty}</span>
                    <span class="percentage-value">${prediction.percentage}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${barWidth}%; background: linear-gradient(90deg, #7c3aed, #8b5cf6);"></div>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    // Create selectable list (all predictions)
    html += `
        <div class="selectable-cfa-list">
            <h5 style="color: #059669; margin: 25px 0 15px 0;">All CFA Predictions (Selectable)</h5>
            <div class="cfa-selection-grid">
    `;

    cfaPredictions.forEach((prediction, index) => {
        html += `
            <div class="cfa-selection-item">
                <label class="cfa-checkbox-label">
                    <input type="checkbox" class="cfa-selection-checkbox" 
                           value="${prediction.specialty}" 
                           id="cfa-${index}">
                    <span class="cfa-specialty-text">${prediction.specialty}</span>
                    <span class="cfa-percentage-badge">${prediction.percentage}%</span>
                </label>
            </div>
        `;
    });

    html += `
            </div>
            <div class="cfa-actions">
                <button id="exportSelectedCfaBtn" class="button secondary-button">
                    Export Selected CFAs
                </button>
                <button id="viewChartBtn" class="button">
                    View Interactive Chart
                </button>
            </div>
        </div>
    `;

    html += `</div>`;
    
    bioCfaOutput.innerHTML = html;
    
    // Add event listeners for new buttons
    document.getElementById('exportSelectedCfaBtn')?.addEventListener('click', handleExportSelectedCfa);
    document.getElementById('viewChartBtn')?.addEventListener('click', handleViewChart);
}

function handleExportSelectedCfa() {
    const checkboxes = document.querySelectorAll('.cfa-selection-checkbox:checked');
    const selectedCfas = Array.from(checkboxes).map(cb => cb.value);
    
    if (selectedCfas.length === 0) {
        alert('Please select at least one CFA specialty.');
        return;
    }
    
    // Create export data
    const exportData = {
        selected_cfas: selectedCfas,
        export_date: new Date().toISOString(),
        pipeline_id: currentPipelineId,
        method: 'bio_clinical_bert_similarity'
    };
    
    // Download as JSON
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bio_clinical_bert_cfas_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function handleViewChart() {
    try {
        const response = await getCfaPredictionChart(currentPipelineId, 15); // Top 15
        
        if (response.chart && response.chart.figure_json) {
            showInteractiveChart(response.chart.figure_json);
        }
        
    } catch (error) {
        alert('Error loading interactive chart: ' + error.message);
    }
}

function showInteractiveChart(figureJson) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.className = 'chart-modal';
    modal.innerHTML = `
        <div class="chart-modal-content">
            <div class="chart-modal-header">
                <h3>Interactive Bio-ClinicalBERT CFA Predictions</h3>
                <button class="close-modal" onclick="this.closest('.chart-modal').remove()">×</button>
            </div>
            <div id="interactiveChart" class="interactive-chart"></div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Render chart
    try {
        const figure = JSON.parse(figureJson);
        Plotly.newPlot('interactiveChart', figure.data, figure.layout, {responsive: true});
    } catch (e) {
        document.getElementById('interactiveChart').innerHTML = `
            <div class="error-message">Error rendering chart</div>
        `;
    }
}

// Utility Functions
function resetUI() {
    // Hide all outputs
    cleanedOutput.classList.add('hidden');
    termsOutput.classList.add('hidden');
    deidentifiedOutput.classList.add('hidden');
    
    // Hide CFA method selection and both CFA sections
    cfaMethodSection.classList.add('hidden');
    llmCfaSection.classList.add('hidden');
    bioClinicalCfaSection.classList.add('hidden');
    
    // Hide all CFA outputs
    llmDiagnosisOutput.classList.add('hidden');
    llmCfaOutput.classList.add('hidden');
    visualizationOutput.classList.add('hidden');
    bioCfaOutput.classList.add('hidden');
    
    // Hide display buttons
    displayCleanedNotesBtn.classList.add('hidden');
    displayExtractedTermsBtn.classList.add('hidden');
    displayDeidentifiedNotesBtn.classList.add('hidden');
    
    // Reset buttons
    predictDiagnosisBtn.disabled = true;
    predictLlmCfaBtn.disabled = true;
    displayVisualizationBtn.disabled = true;
    predictBioCfaBtn.disabled = true;
    
    // Reset CFA method selection
    selectedCfaMethod = null;
    llmCfaBtn.classList.remove('active');
    bioClinicalCfaBtn.classList.remove('active');
    
    // Reset progress bar
    progressFill.style.width = '0%';
    progressFill.style.backgroundColor = '#1e40af';
    
    // Clear any existing intervals
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    if (diagnosisStatusInterval) clearInterval(diagnosisStatusInterval);
    if (cfaStatusInterval) clearInterval(cfaStatusInterval);
    if (bioCfaStatusInterval) clearInterval(bioCfaStatusInterval);
}

function resetButtons() {
    startPipelineBtn.disabled = false;
    startPipelineBtn.textContent = 'Start Processing';
}

function showError(message) {
    statusText.textContent = message;
    progressFill.style.backgroundColor = '#dc3545';
}

// API functions - Core Pipeline
async function startPipeline(soapNotes, location) {
    const response = await fetch(`${API_BASE_URL}/start-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            soapNotes: soapNotes,
            location: location
        })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getPipelineStatus(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/pipeline-status/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getDeidentifiedNotes(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-deidentified-notes/${pipelineId}`);

    if (response.status === 202) {
        throw new Error('De-identified notes not ready yet');
    }

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getCleanedNotes(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-cleaned-notes/${pipelineId}`);

    if (response.status === 202) {
        throw new Error('Cleaned notes not ready yet');
    }

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getExtractedTerms(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-extracted-terms/${pipelineId}`);

    if (response.status === 202) {
        throw new Error('Term extraction not ready yet');
    }

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

// LLM API functions
async function startDiagnosisPrediction(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/predict-diagnosis/${pipelineId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getDiagnosisStatus(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/diagnosis-status/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getDiagnoses(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-diagnoses/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function startCfaPrediction(pipelineId, selectedDiagnoses) {
    const response = await fetch(`${API_BASE_URL}/predict-cfa/${pipelineId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_diagnoses: selectedDiagnoses })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getCfaStatus(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/cfa-status/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getCfaPredictions(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-cfa-predictions/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

// Bio-ClinicalBERT API functions
async function predictSimilarityCfa(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/predict-similarity-cfa/${pipelineId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getSimilarityCfaStatus(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/similarity-cfa-status/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getSimilarityCfaPredictions(pipelineId) {
    const response = await fetch(`${API_BASE_URL}/get-similarity-cfa-predictions/${pipelineId}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

async function getCfaVisualization(pipelineId, method = 'tsne') {
    console.log(`🌐 Calling getCfaVisualization(${pipelineId}, ${method})`);
    
    const response = await fetch(`${API_BASE_URL}/get-cfa-visualization/${pipelineId}?method=${method}`);
    
    console.log('🔍 Response status:', response.status);
    console.log('🔍 Response headers:', response.headers);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ API error:', errorData);
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ API response data:', data);
    return data;
}

async function getCfaPredictionChart(pipelineId, topN = 10) {
    const response = await fetch(`${API_BASE_URL}/get-cfa-prediction-chart/${pipelineId}?top_n=${topN}`);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

// Health check function
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('Backend health:', data);
        
        if (!data.medical_nlp_ready) {
            console.warn('Medical NLP processor not ready');
            showHealthWarning('Medical NLP processor not ready. Some features may not work.');
        }
        
        if (!data.cfa_engine_ready) {
            console.warn('CFA engine not ready');
            showHealthWarning('Bio-ClinicalBERT CFA engine not ready. Similarity analysis may not work.');
        }
        
        if (data.medical_nlp_ready && data.cfa_engine_ready) {
            console.log('✅ All systems ready!');
        }
        
    } catch (error) {
        console.error('Backend health check failed:', error);
        showHealthWarning('Backend connection failed. Please check if the server is running.');
    }
}

function showHealthWarning(message) {
    // Create a temporary warning banner
    const warning = document.createElement('div');
    warning.className = 'health-warning';
    warning.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #f59e0b;
        color: white;
        padding: 10px;
        text-align: center;
        font-weight: 600;
        z-index: 1000;
        animation: slideDown 0.3s ease;
    `;
    warning.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="margin-left: 15px; background: none; border: 1px solid white; color: white; padding: 2px 8px; border-radius: 3px; cursor: pointer;">×</button>
    `;
    
    document.body.insertBefore(warning, document.body.firstChild);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (warning.parentElement) {
            warning.remove();
        }
    }, 10000);
}

// Add CSS for health warning animation and modal
const style = document.createElement('style');
style.textContent = `
    @keyframes slideDown {
        from { transform: translateY(-100%); }
        to { transform: translateY(0); }
    }
    
    .chart-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 2000;
    }
    
    .chart-modal-content {
        background: white;
        border-radius: 12px;
        width: 90%;
        max-width: 1200px;
        height: 80%;
        display: flex;
        flex-direction: column;
    }
    
    .chart-modal-header {
        padding: 20px;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .interactive-chart {
        flex: 1;
        padding: 20px;
    }
    
    .close-modal {
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        color: #6b7280;
    }
`;
document.head.appendChild(style);

// Initialize application
function initializeApp() {
    console.log('🚀 Medical NLP Pipeline with Bio-ClinicalBERT initialized');
    if (typeof Plotly !== 'undefined') {
        console.log('✅ Plotly.js loaded successfully, version:', Plotly.version);
    } else {
        console.error('❌ Plotly.js not loaded! Charts will not work.');
        showHealthWarning('Plotly.js library not loaded. Visualizations will not work.');
    }
    // Check backend health
    checkBackendHealth();
    
    // Set up periodic health checks (every 5 minutes)
    setInterval(checkBackendHealth, 5 * 60 * 1000);
    
    // Add sample SOAP notes for testing
}

// Check backend health on page load and initialize
window.addEventListener('load', initializeApp);

// Handle page unload - cleanup intervals
window.addEventListener('beforeunload', () => {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    if (diagnosisStatusInterval) clearInterval(diagnosisStatusInterval);
    if (cfaStatusInterval) clearInterval(cfaStatusInterval);
    if (bioCfaStatusInterval) clearInterval(bioCfaStatusInterval);
});

// Add keyboard shortcuts for power users
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to start pipeline
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!startPipelineBtn.disabled) {
            handleStartPipeline();
        }
        e.preventDefault();
    }
    
    // Ctrl/Cmd + Shift + C to clear all
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        if (confirm('Clear all data and reset the pipeline?')) {
            soapNotesInput.value = '';
            locationInput.value = '';
            resetUI();
            currentPipelineId = null;
        }
        e.preventDefault();
    }
});

// Add tooltip for keyboard shortcuts
const shortcutTooltip = document.createElement('div');
shortcutTooltip.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    font-size: 12px;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
`;
shortcutTooltip.innerHTML = `
    <div><strong>Keyboard Shortcuts:</strong></div>
    <div>Ctrl+Enter: Start Pipeline</div>
    <div>Ctrl+Shift+C: Clear All</div>
`;
document.body.appendChild(shortcutTooltip);

// Show/hide shortcuts tooltip
let tooltipTimeout;
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        clearTimeout(tooltipTimeout);
        shortcutTooltip.style.opacity = '1';
    }
});

document.addEventListener('keyup', (e) => {
    if (!e.ctrlKey && !e.metaKey) {
        tooltipTimeout = setTimeout(() => {
            shortcutTooltip.style.opacity = '0';
        }, 1000);
    }
});