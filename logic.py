import numpy as np
import pandas as pd
import chromadb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Tuple, Any
import json

# Import your existing functions
from db import create_embeddings, load_model
from medical_nlp import extract_priority_terms_list

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def convert_numpy_to_python(obj):
    """
    Recursively convert NumPy arrays and types to native Python types for JSON serialization
    
    Args:
        obj: Object that may contain NumPy arrays or types
        
    Returns:
        Object with NumPy arrays converted to Python lists and types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj

class CFAPredictionEngine:
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize the CFA prediction engine"""
        self.chroma_db_path = chroma_db_path
        self.client = None
        self.collection = None
        self.cfa_terms = []
        self.cfa_embeddings = []
        self.knn_model = None
        self._setup_chromadb()
        self._setup_knn()
        
    def _setup_chromadb(self):
        """Setup ChromaDB connection and load CFA terms"""
        try:
            logger.info("Connecting to ChromaDB...")
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = self.client.get_collection(name="cfa_terms")
            
            # Get all CFA terms and embeddings
            results = self.collection.get(include=['documents', 'embeddings', 'metadatas'])
            self.cfa_terms = results['documents']
            self.cfa_embeddings = np.array(results['embeddings'])
            
            logger.info(f"Loaded {len(self.cfa_terms)} CFA terms from database")
            
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            raise
    
    def _setup_knn(self):
        """Setup KNN model for similarity search"""
        try:
            logger.info("Setting up KNN model...")
            # Use cosine similarity for medical terms
            self.knn_model = NearestNeighbors(
                n_neighbors=3,  # Top 3 neighbors
                metric='cosine',
                algorithm='brute'  # Better for high-dimensional data
            )
            self.knn_model.fit(self.cfa_embeddings)
            logger.info("KNN model setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up KNN model: {e}")
            raise
    
    def get_priority_terms_embeddings(self, priority_terms: List[str]) -> np.ndarray:
        """
        Create embeddings for priority terms using Bio-ClinicalBERT
        
        Args:
            priority_terms: List of priority medical terms
            
        Returns:
            np.ndarray: Embeddings for priority terms
        """
        try:
            logger.info(f"Creating embeddings for {len(priority_terms)} priority terms")
            
            if not priority_terms:
                logger.warning("No priority terms provided")
                return np.array([])
            
            # Ensure model is loaded
            load_model()
            
            # Create embeddings
            embeddings = create_embeddings(priority_terms)
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating priority term embeddings: {e}")
            raise
    
    def find_top_3_cfa_for_each_term_knn(self, priority_terms: List[str], priority_embeddings: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find top 3 CFA terms for each priority term using KNN
        
        Args:
            priority_terms: List of priority medical terms
            priority_embeddings: Embeddings of priority terms
            
        Returns:
            Dict mapping each priority term to its top 3 CFA matches with similarity scores
        """
        try:
            logger.info("Finding top 3 CFA terms for each priority term using KNN...")
            
            if len(priority_embeddings) == 0:
                return {}
            
            # Find k-nearest neighbors for each priority term
            distances, indices = self.knn_model.kneighbors(priority_embeddings)
            
            # Convert cosine distances to similarity scores (1 - distance)
            similarities = 1 - distances
            
            results = {}
            
            for i, priority_term in enumerate(priority_terms):
                term_matches = []
                
                for j in range(3):  # Top 3
                    cfa_idx = indices[i][j]
                    cfa_term = self.cfa_terms[cfa_idx]
                    similarity_score = similarities[i][j]
                    
                    term_matches.append((cfa_term, float(similarity_score)))
                
                results[priority_term] = term_matches
            
            logger.info(f"Found top 3 CFA matches for {len(results)} priority terms")
            return results
            
        except Exception as e:
            logger.error(f"Error finding top 3 CFA terms: {e}")
            raise
    
    def calculate_similarity_scores(self, priority_embeddings: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calculate similarity scores between priority terms and CFA terms (original method)
        
        Args:
            priority_embeddings: Embeddings of priority terms
            
        Returns:
            Dict mapping each priority term to its top CFA matches with scores
        """
        try:
            logger.info("Calculating similarity scores...")
            
            if len(priority_embeddings) == 0:
                return {}
            
            # Calculate cosine similarities
            # Normalize embeddings for cosine similarity
            priority_norm = priority_embeddings / np.linalg.norm(priority_embeddings, axis=1, keepdims=True)
            cfa_norm = self.cfa_embeddings / np.linalg.norm(self.cfa_embeddings, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(priority_norm, cfa_norm.T)
            
            # For each priority term, get top 3 CFA matches
            results = {}
            for i, priority_embedding in enumerate(priority_embeddings):
                similarities = similarity_matrix[i]
                
                # Get top 3 indices
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                # Store results
                term_matches = []
                for idx in top_indices:
                    cfa_term = self.cfa_terms[idx]
                    score = similarities[idx]
                    term_matches.append((cfa_term, float(score)))
                
                results[f"priority_term_{i}"] = term_matches
            
            logger.info(f"Calculated similarities for {len(results)} priority terms")
            return results
            
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            raise
    
    def calculate_cfa_percentages_from_knn(self, knn_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        """
        Calculate percentage distribution of CFA specialties based on KNN weighted similarity scores
        
        Args:
            knn_results: Results from KNN search
            
        Returns:
            Dict mapping CFA specialties to their percentage scores
        """
        try:
            logger.info("Calculating CFA specialty percentages from KNN results...")
            
            # Aggregate scores by CFA specialty with weights
            cfa_scores = {}
            
            for priority_term, matches in knn_results.items():
                for rank, (cfa_term, similarity) in enumerate(matches):
                    # Weight by rank: 1st = 3 points, 2nd = 2 points, 3rd = 1 point
                    weight = 3 - rank
                    weighted_score = similarity * weight
                    
                    if cfa_term not in cfa_scores:
                        cfa_scores[cfa_term] = 0
                    
                    cfa_scores[cfa_term] += weighted_score
            
            # Calculate percentages
            total_score = sum(cfa_scores.values())
            cfa_percentages = {}
            
            if total_score > 0:
                # Sort by score (descending)
                sorted_cfa = sorted(cfa_scores.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (cfa_term, score) in enumerate(sorted_cfa, 1):
                    percentage = (score / total_score) * 100
                    cfa_percentages[cfa_term] = round(percentage, 2)
            
            logger.info(f"Calculated percentages for {len(cfa_percentages)} CFA specialties")
            return cfa_percentages
            
        except Exception as e:
            logger.error(f"Error calculating CFA percentages: {e}")
            raise
    
    def predict_cfa_specialties(self, priority_terms: List[str], use_knn: bool = True) -> Dict[str, Any]:
        """
        Predict CFA specialties based on priority terms using KNN or traditional similarity scoring
        
        Args:
            priority_terms: List of priority medical terms
            use_knn: Whether to use KNN approach (default: True)
            
        Returns:
            Dict containing predictions, scores, and metadata
        """
        try:
            method_name = "KNN" if use_knn else "Traditional Similarity"
            logger.info(f"Predicting CFA specialties for {len(priority_terms)} priority terms using {method_name}")
            
            if not priority_terms:
                return {
                    "cfa_predictions": [],
                    "cfa_percentages": {},
                    "knn_results": {} if use_knn else None,
                    "similarity_details": {} if not use_knn else None,
                    "total_terms": 0,
                    "method": method_name,
                    "message": "No priority terms provided"
                }
            
            # Get embeddings for priority terms
            priority_embeddings = self.get_priority_terms_embeddings(priority_terms)
            
            if len(priority_embeddings) == 0:
                return {
                    "cfa_predictions": [],
                    "cfa_percentages": {},
                    "knn_results": {} if use_knn else None,
                    "similarity_details": {} if not use_knn else None,
                    "total_terms": 0,
                    "method": method_name,
                    "message": "Could not create embeddings for priority terms"
                }
            
            # Choose method for finding CFA matches
            if use_knn:
                # Use KNN approach
                similarity_results = self.find_top_3_cfa_for_each_term_knn(priority_terms, priority_embeddings)
                cfa_percentages = self.calculate_cfa_percentages_from_knn(similarity_results)
                knn_results = similarity_results
                similarity_details = None
            else:
                # Use traditional similarity approach
                similarity_results = self.calculate_similarity_scores(priority_embeddings)
                # Aggregate scores by CFA specialty (original method)
                cfa_scores = {}
                for term_key, matches in similarity_results.items():
                    for cfa_term, score in matches:
                        if cfa_term not in cfa_scores:
                            cfa_scores[cfa_term] = []
                        cfa_scores[cfa_term].append(score)
                
                # Calculate weighted scores (sum of similarities)
                cfa_weighted_scores = {}
                for cfa_term, scores in cfa_scores.items():
                    cfa_weighted_scores[cfa_term] = sum(scores)
                
                # Normalize to percentages
                total_score = sum(cfa_weighted_scores.values())
                cfa_percentages = {}
                
                if total_score > 0:
                    for cfa_term, score in cfa_weighted_scores.items():
                        percentage = (score / total_score) * 100
                        cfa_percentages[cfa_term] = round(percentage, 1)
                
                knn_results = None
                similarity_details = similarity_results
            
            # Create final predictions list
            cfa_predictions = []
            for cfa_term, percentage in sorted(cfa_percentages.items(), key=lambda x: x[1], reverse=True):
                cfa_predictions.append({
                    "specialty": cfa_term,
                    "percentage": percentage
                })
            
            logger.info(f"Generated {len(cfa_predictions)} CFA predictions using {method_name}")
            
            return {
                "cfa_predictions": cfa_predictions,
                "cfa_percentages": cfa_percentages,
                "knn_results": knn_results,
                "similarity_details": similarity_details,
                "priority_terms": priority_terms,
                "total_terms": len(priority_terms),
                "method": method_name,
                "message": f"Successfully predicted {len(cfa_predictions)} CFA specialties using {method_name}"
            }
            
        except Exception as e:
            logger.error(f"Error predicting CFA specialties: {e}")
            raise
    
    def create_2d_visualization(self, priority_terms: List[str], method: str = "tsne") -> Dict[str, Any]:
        """
        Create 2D visualization of priority terms and CFA terms
        
        Args:
            priority_terms: List of priority medical terms
            method: Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            Dict containing plot data and metadata
        """
        try:
            logger.info(f"Creating 2D visualization using {method.upper()}")
            
            if not priority_terms:
                return {"error": "No priority terms provided"}
            
            # Get priority term embeddings
            priority_embeddings = self.get_priority_terms_embeddings(priority_terms)
            
            if len(priority_embeddings) == 0:
                return {"error": "Could not create embeddings for priority terms"}
            
            # Combine all embeddings
            all_embeddings = np.vstack([priority_embeddings, self.cfa_embeddings])
            all_labels = (["Priority"] * len(priority_terms) + 
                         ["CFA"] * len(self.cfa_terms))
            all_texts = priority_terms + self.cfa_terms
            
            # Apply dimensionality reduction
            if method.lower() == "pca":
                reducer = PCA(n_components=2, random_state=42)
            else:  # tsne
                perplexity = min(30, len(all_embeddings) - 1, 50)
                reducer = TSNE(
                    n_components=2, 
                    random_state=42, 
                    perplexity=perplexity,
                    n_iter=1000,
                    learning_rate='auto'
                )
            
            embeddings_2d = reducer.fit_transform(all_embeddings)
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'label': all_labels,
                'text': all_texts,
                'type': ['Priority Term' if label == 'Priority' else 'CFA Term' for label in all_labels]
            })
            
            logger.info(f"Created 2D visualization with {len(plot_data)} points")
            
            return {
                "plot_data": plot_data.to_dict('records'),
                "method": method,
                "n_priority": len(priority_terms),
                "n_cfa": len(self.cfa_terms),
                "message": f"Successfully created 2D visualization using {method.upper()}"
            }
            
        except Exception as e:
            logger.error(f"Error creating 2D visualization: {e}")
            return {"error": f"Failed to create visualization: {str(e)}"}
    
    def create_plotly_visualization(self, priority_terms: List[str], method: str = "tsne") -> Dict[str, Any]:
        """
        Create interactive Plotly visualization
        
        Args:
            priority_terms: List of priority medical terms
            method: Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            Dict containing Plotly figure JSON and metadata
        """
        try:
            viz_data = self.create_2d_visualization(priority_terms, method)
            
            if "error" in viz_data:
                return viz_data
            
            plot_data = pd.DataFrame(viz_data["plot_data"])
            
            # Create Plotly scatter plot with enhanced styling
            fig = go.Figure()
            
            # Color scheme
            colors = {
                'Priority Term': '#FF6B6B',  # Red for priority terms
                'CFA Term': '#4ECDC4'        # Teal for CFA terms
            }
            
            # Add traces for each term type
            for term_type in ['Priority Term', 'CFA Term']:
                mask = plot_data['type'] == term_type
                data_subset = plot_data[mask]
                
                # Convert NumPy arrays to Python lists
                x_coords = convert_numpy_to_python(data_subset['x'].values)
                y_coords = convert_numpy_to_python(data_subset['y'].values)
                text_labels = convert_numpy_to_python(data_subset['text'].values)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            size=12 if term_type == 'Priority Term' else 8,
                            color=colors[term_type],
                            opacity=0.8,
                            line=dict(width=2, color='white') if term_type == 'Priority Term' else dict(width=1, color='white'),
                            symbol='diamond' if term_type == 'Priority Term' else 'circle'
                        ),
                        text=text_labels,
                        name=term_type,
                        hovertemplate=f'<b>%{{text}}</b><br>Type: {term_type}<br>Method: {method.upper()}<extra></extra>',
                        showlegend=True
                    )
                )
            
            # Update layout
            method_title = "Principal Component Analysis" if method.lower() == "pca" else "t-SNE"
            x_label = "PC1" if method.lower() == "pca" else "t-SNE 1"
            y_label = "PC2" if method.lower() == "pca" else "t-SNE 2"
            
            fig.update_layout(
                title=f'{method_title} Visualization<br><sub>Priority Terms vs CFA Medical Specialties</sub>',
                xaxis_title=x_label,
                yaxis_title=y_label,
                width=800,
                height=600,
                hovermode='closest',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                plot_bgcolor='rgba(248,249,250,0.8)',
                paper_bgcolor='white'
            )
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            
            # Convert to Python types for safe JSON serialization
            figure_dict = convert_numpy_to_python(fig.to_dict())
            
            return {
                "figure_json": json.dumps(figure_dict),
                "method": method,
                "n_priority": viz_data["n_priority"],
                "n_cfa": viz_data["n_cfa"],
                "message": viz_data["message"]
            }
            
        except Exception as e:
            logger.error(f"Error creating Plotly visualization: {e}")
            return {"error": f"Failed to create Plotly visualization: {str(e)}"}
    
    def create_cfa_prediction_chart(self, cfa_predictions: List[Dict], top_n: int = 10) -> Dict[str, Any]:
        """
        Create chart for CFA predictions
        
        Args:
            cfa_predictions: List of CFA prediction dictionaries
            top_n: Number of top predictions to show
            
        Returns:
            Dict containing chart data and Plotly figure JSON
        """
        try:
            logger.info(f"Creating CFA prediction chart for top {top_n} predictions")
            
            if not cfa_predictions:
                return {"error": "No CFA predictions provided"}
            
            # Get top N predictions
            top_predictions = cfa_predictions[:top_n]
            
            # Create DataFrame
            chart_data = pd.DataFrame(top_predictions)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Convert data to Python types
            percentages = convert_numpy_to_python(chart_data['percentage'].values)
            specialties = convert_numpy_to_python(chart_data['specialty'].values)
            
            fig.add_trace(go.Bar(
                x=percentages,
                y=specialties,
                orientation='h',
                marker=dict(
                    color=percentages,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Percentage (%)",
                        titleside="right"
                    )
                ),
                text=[f"{p:.1f}%" for p in percentages],
                textposition='inside',
                textfont=dict(color='white', size=11, family="Arial Black"),
                hovertemplate='<b>%{y}</b><br>Percentage: %{x:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Top {len(top_predictions)} CFA Specialty Predictions<br><sub>Based on KNN Similarity and Weighted Scoring</sub>',
                xaxis_title='Prediction Confidence (%)',
                yaxis_title='Medical Specialty',
                height=max(500, len(top_predictions) * 35),
                width=1000,
                yaxis=dict(
                    autorange="reversed",  # Show highest percentage at top
                    tickfont=dict(size=10)
                ),
                xaxis=dict(
                    tickformat='.1f',
                    ticksuffix='%'
                ),
                showlegend=False,
                margin=dict(l=200, r=100, t=80, b=60)
            )
            
            # Convert entire figure to Python types
            figure_dict = convert_numpy_to_python(fig.to_dict())
            
            return {
                "chart_data": convert_numpy_to_python(chart_data.to_dict('records')),
                "figure_json": json.dumps(figure_dict),
                "top_n": len(top_predictions),
                "total_predictions": len(cfa_predictions),
                "message": f"Created chart for top {len(top_predictions)} CFA predictions"
            }
            
        except Exception as e:
            logger.error(f"Error creating CFA prediction chart: {e}")
            return {"error": f"Failed to create prediction chart: {str(e)}"}

# Global instance
cfa_engine = None

def get_cfa_engine() -> CFAPredictionEngine:
    """Get or create the CFA prediction engine instance"""
    global cfa_engine
    if cfa_engine is None:
        cfa_engine = CFAPredictionEngine()
    return cfa_engine

def run_complete_cfa_analysis(priority_terms: List[str], use_knn: bool = True) -> Dict[str, Any]:
    """
    Run complete CFA analysis including KNN, predictions, and visualizations
    
    Args:
        priority_terms: List of priority medical terms
        use_knn: Whether to use KNN approach (default: True)
        
    Returns:
        Dict containing all analysis results
    """
    try:
        method_name = "KNN" if use_knn else "Traditional Similarity"
        
        engine = get_cfa_engine()
        
        # 1. Get CFA predictions using selected method
        prediction_results = engine.predict_cfa_specialties(priority_terms, use_knn=use_knn)
        
        # 2. Create prediction chart
        chart_results = engine.create_cfa_prediction_chart(
            prediction_results.get("cfa_predictions", []), 
            top_n=15
        )
        
        # 3. Create 2D visualizations
        pca_viz = engine.create_plotly_visualization(priority_terms, "pca")
        tsne_viz = engine.create_plotly_visualization(priority_terms, "tsne")
        
        result = {
            "pca_visualization": pca_viz,
            "tsne_visualization": tsne_viz,
            "n_priority": len(priority_terms),
            "n_cfa": len(engine.cfa_terms),
            "message": "Successfully created both PCA and t-SNE visualizations"
        }
        
        # Convert result to Python types
        return convert_numpy_to_python(result)
        
    except Exception as e:
        logger.error(f"Error creating both visualizations: {e}")
        return {"error": f"Both visualizations creation failed: {str(e)}"}

def export_selected_cfas(selected_cfas: List[str], pipeline_id: str = None) -> Dict[str, Any]:
    """
    Export selected CFA specialties to JSON format
    
    Args:
        selected_cfas: List of selected CFA specialty names
        pipeline_id: Optional pipeline identifier
        
    Returns:
        Dict containing export data
    """
    try:
        export_data = {
            "selected_cfas": selected_cfas,
            "export_date": pd.Timestamp.now().isoformat(),
            "pipeline_id": pipeline_id,
            "method": "bio_clinical_bert_knn_similarity",
            "total_selected": len(selected_cfas),
            "metadata": {
                "analysis_type": "CFA_specialty_prediction",
                "embedding_model": "Bio-ClinicalBERT",
                "similarity_method": "KNN_cosine_similarity",
                "ranking_method": "weighted_similarity_scores"
            }
        }
        
        logger.info(f"Exported {len(selected_cfas)} selected CFA specialties")
        result = {
            "export_data": export_data,
            "success": True,
            "message": f"Successfully exported {len(selected_cfas)} CFA specialties"
        }
        
        # ✅ CRITICAL FIX: Convert result to Python types
        return convert_numpy_to_python(result)
        
    except Exception as e:
        logger.error(f"Error exporting selected CFAs: {e}")
        return {"error": f"Export failed: {str(e)}"}

def predict_cfa_from_priority_terms(priority_terms: List[str]) -> Dict[str, Any]:
    """
    Main function to predict CFA specialties from priority terms using KNN
    
    Args:
        priority_terms: List of priority medical terms
        
    Returns:
        Dict containing predictions and metadata
    """
    try:
        engine = get_cfa_engine()
        result = engine.predict_cfa_specialties(priority_terms, use_knn=True)
        # Convert result to Python types
        return convert_numpy_to_python(result)
    except Exception as e:
        logger.error(f"Error in CFA prediction: {e}")
        return {"error": f"CFA prediction failed: {str(e)}"}

def create_visualization_data(priority_terms: List[str], method: str = "tsne") -> Dict[str, Any]:
    """
    Create visualization data for priority terms and CFA terms
    
    Args:
        priority_terms: List of priority medical terms
        method: Dimensionality reduction method
        
    Returns:
        Dict containing visualization data
    """
    try:
        engine = get_cfa_engine()
        result = engine.create_plotly_visualization(priority_terms, method)
        # Convert result to Python types
        return convert_numpy_to_python(result)
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return {"error": f"Visualization creation failed: {str(e)}"}

def create_prediction_chart_data(cfa_predictions: List[Dict], top_n: int = 10) -> Dict[str, Any]:
    """
    Create chart data for CFA predictions
    
    Args:
        cfa_predictions: List of CFA prediction dictionaries
        top_n: Number of top predictions to show
        
    Returns:
        Dict containing chart data
    """
    try:
        engine = get_cfa_engine()
        result = engine.create_cfa_prediction_chart(cfa_predictions, top_n)
        # Convert result to Python types
        return convert_numpy_to_python(result)
    except Exception as e:
        logger.error(f"Error creating prediction chart: {e}")
        return {"error": f"Chart creation failed: {str(e)}"}

def get_selectable_cfa_list(cfa_predictions: List[Dict]) -> List[Dict[str, Any]]:
    """
    Get CFA predictions formatted for selectable list UI
    
    Args:
        cfa_predictions: List of CFA prediction dictionaries
        
    Returns:
        List of dictionaries formatted for UI selection
    """
    try:
        selectable_list = []
        
        for pred in cfa_predictions:
            selectable_list.append({
                "id": f"cfa_{len(selectable_list)}",
                "specialty": pred["specialty"],
                "percentage": pred["percentage"],
                "display_text": f"{pred['specialty']} ({pred['percentage']}%)",
                "selected": False
            })
        
        # Convert result to Python types
        return convert_numpy_to_python(selectable_list)
        
    except Exception as e:
        logger.error(f"Error creating selectable CFA list: {e}")
        return []

def create_both_visualizations(priority_terms: List[str]) -> Dict[str, Any]:
    """
    Create both PCA and t-SNE visualizations separately
    
    Args:
        priority_terms: List of priority medical terms
        
    Returns:
        Dict containing both separate visualizations
    """
    try:
        engine = get_cfa_engine()        
        # Create both visualizations
        pca_viz = engine.create_plotly_visualization(priority_terms, "pca")
        tsne_viz = engine.create_plotly_visualization(priority_terms, "tsne")
        
        result = {
            "pca_visualization": pca_viz,
            "tsne_visualization": tsne_viz,
            "n_priority": len(priority_terms),
            "n_cfa": len(engine.cfa_terms),
            "message": "Successfully created both PCA and t-SNE visualizations"
        }
        

        return convert_numpy_to_python(result)       
    except Exception as e:
        logger.error(f"Error creating both visualizations: {e}")
        return {"error": f"Both visualizations creation failed: {str(e)}"}

def display_knn_analysis_summary(analysis_results: Dict[str, Any]) -> None:
    """
    Display a comprehensive summary of the KNN analysis results
    
    Args:
        analysis_results: Results from run_complete_cfa_analysis
    """
    try:
        # Extract key information
        predictions = analysis_results.get("predictions", {})
        summary = analysis_results.get("summary", {})
        selectable_cfas = analysis_results.get("selectable_cfas", [])

        # Optionally, you could return this as a structured dict instead of printing
        return {
            "method": summary.get("method_used", "Unknown"),
            "total_priority_terms": summary.get("total_priority_terms", 0),
            "total_cfa_terms": summary.get("total_cfa_terms", 0),
            "cfa_predictions_top5": predictions.get("cfa_predictions", [])[:5],
            "total_selectable_cfas": len(selectable_cfas),
            "knn_summary": {
                "terms_analyzed": len(predictions.get("knn_results", {})),
                "matches_per_term": 3,
                "ranking_weights": "1st=3pts, 2nd=2pts, 3rd=1pt"
            } if predictions.get("knn_results") else None,
            "analysis_complete": summary.get("analysis_complete", False)
        }
        
    except Exception as e:
        logger.error(f"Error displaying analysis summary: {e}")
