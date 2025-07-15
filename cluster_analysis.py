"""
Cluster-based Material Analysis System
Uses K-means clustering for regional material comparison and advanced uniformity assessment
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from typing import Dict, List, Tuple, Any
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy import ndimage
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterMaterialAnalyzer:
    """
    Advanced material analysis using K-means clustering for regional comparison
    """
    
    def __init__(self, n_clusters: int = 4, target_size: Tuple[int, int] = (400, 400)):
        """
        Initialize the cluster analyzer
        
        Args:
            n_clusters: Number of clusters for K-means analysis
            target_size: Target size for image standardization
        """
        self.n_clusters = n_clusters
        self.target_size = target_size
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for cluster analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Resize to standard dimensions for consistent analysis
            image = cv2.resize(image, self.target_size)
            
            # Apply slight Gaussian blur to reduce noise
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            logger.info(f"Loaded and preprocessed image: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def extract_color_clusters(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract color clusters using K-means clustering in LAB color space
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary containing cluster information
        """
        try:
            # Convert to LAB color space for perceptually uniform clustering
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Reshape image to pixel array
            pixels = lab_image.reshape((-1, 3))
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Reshape labels back to image dimensions
            label_map = labels.reshape(image.shape[:2])
            
            # Get cluster centers (in LAB color space)
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(self.n_clusters):
                cluster_mask = (label_map == i)
                cluster_size = np.sum(cluster_mask)
                cluster_percentage = (cluster_size / label_map.size) * 100
                
                cluster_stats.append({
                    'cluster_id': i,
                    'center_lab': cluster_centers[i].tolist(),
                    'size_pixels': int(cluster_size),
                    'percentage': float(cluster_percentage)
                })
            
            # Calculate silhouette score for cluster quality
            silhouette_avg = silhouette_score(pixels, labels)
            
            return {
                'cluster_centers': cluster_centers.tolist(),
                'label_map': label_map.tolist(),
                'cluster_stats': cluster_stats,
                'silhouette_score': float(silhouette_avg),
                'n_clusters': self.n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in color clustering: {e}")
            raise
    
    def extract_texture_clusters(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract texture features for each color cluster
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary containing texture information per cluster
        """
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get color clusters first
            color_clusters = self.extract_color_clusters(image)
            label_map = np.array(color_clusters['label_map'])
            
            texture_features = []
            
            for cluster_id in range(self.n_clusters):
                cluster_mask = (label_map == cluster_id)
                
                if np.sum(cluster_mask) < 50:  # Skip very small clusters
                    texture_features.append({
                        'cluster_id': cluster_id,
                        'lbp_histogram': [],
                        'glcm_contrast': 0.0,
                        'glcm_dissimilarity': 0.0,
                        'glcm_homogeneity': 0.0,
                        'gabor_mean': 0.0,
                        'gabor_std': 0.0
                    })
                    continue
                
                # Extract region of interest
                cluster_region = gray * cluster_mask
                
                # Local Binary Pattern (LBP)
                lbp = local_binary_pattern(cluster_region, P=8, R=1, method='uniform')
                lbp_hist, _ = np.histogram(lbp[cluster_mask], bins=10, range=(0, 10))
                lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
                
                # Gray Level Co-occurrence Matrix (GLCM)
                try:
                    glcm = graycomatrix(
                        cluster_region.astype(np.uint8), 
                        distances=[1], 
                        angles=[0], 
                        levels=256,
                        symmetric=True, 
                        normed=True
                    )
                    
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                except:
                    contrast = dissimilarity = homogeneity = 0.0
                
                # Gabor filter response
                try:
                    gabor_response, _ = gabor(cluster_region, frequency=0.6)
                    gabor_mean = np.mean(gabor_response[cluster_mask])
                    gabor_std = np.std(gabor_response[cluster_mask])
                except:
                    gabor_mean = gabor_std = 0.0
                
                texture_features.append({
                    'cluster_id': cluster_id,
                    'lbp_histogram': lbp_hist.tolist(),
                    'glcm_contrast': float(contrast),
                    'glcm_dissimilarity': float(dissimilarity),
                    'glcm_homogeneity': float(homogeneity),
                    'gabor_mean': float(gabor_mean),
                    'gabor_std': float(gabor_std)
                })
            
            return {
                'texture_features': texture_features,
                'analysis_method': 'cluster_based_texture'
            }
            
        except Exception as e:
            logger.error(f"Error in texture clustering: {e}")
            raise
    
    def calculate_cluster_uniformity(self, original_clusters: Dict, replacement_clusters: Dict) -> Dict[str, Any]:
        """
        Calculate uniformity between corresponding clusters of two images
        
        Args:
            original_clusters: Cluster data from original image
            replacement_clusters: Cluster data from replacement image
            
        Returns:
            Uniformity analysis results
        """
        try:
            cluster_comparisons = []
            total_delta_e = 0.0
            total_texture_diff = 0.0
            
            # Compare each cluster pair
            for i in range(min(self.n_clusters, len(original_clusters['cluster_stats']))):
                orig_center = original_clusters['cluster_centers'][i]
                repl_center = replacement_clusters['cluster_centers'][i]
                
                # Calculate Delta-E between cluster centers
                orig_lab = LabColor(lab_l=orig_center[0], lab_a=orig_center[1], lab_b=orig_center[2])
                repl_lab = LabColor(lab_l=repl_center[0], lab_a=repl_center[1], lab_b=repl_center[2])
                
                delta_e = delta_e_cie2000(orig_lab, repl_lab)
                
                # Calculate size difference
                orig_size = original_clusters['cluster_stats'][i]['percentage']
                repl_size = replacement_clusters['cluster_stats'][i]['percentage']
                size_diff = abs(orig_size - repl_size)
                
                # Assess cluster uniformity
                if delta_e < 2.3:
                    uniformity_grade = 'Excellent'
                elif delta_e < 5.0:
                    uniformity_grade = 'Good'
                elif delta_e < 10.0:
                    uniformity_grade = 'Acceptable'
                else:
                    uniformity_grade = 'Poor'
                
                cluster_comparisons.append({
                    'cluster_id': i,
                    'delta_e': float(delta_e),
                    'size_difference_percent': float(size_diff),
                    'original_percentage': float(orig_size),
                    'replacement_percentage': float(repl_size),
                    'uniformity_grade': uniformity_grade
                })
                
                total_delta_e += delta_e
                
            # Calculate overall metrics
            avg_delta_e = total_delta_e / len(cluster_comparisons) if cluster_comparisons else 0
            
            # Overall assessment
            if avg_delta_e < 2.3:
                overall_assessment = 'Uniform'
                confidence = 95
            elif avg_delta_e < 5.0:
                overall_assessment = 'Mostly Uniform'
                confidence = 80
            elif avg_delta_e < 10.0:
                overall_assessment = 'Moderately Uniform'
                confidence = 60
            else:
                overall_assessment = 'Non-Uniform'
                confidence = 40
            
            return {
                'cluster_comparisons': cluster_comparisons,
                'average_delta_e': float(avg_delta_e),
                'overall_assessment': overall_assessment,
                'confidence_percentage': confidence,
                'analysis_method': 'k_means_cluster_comparison',
                'clusters_analyzed': len(cluster_comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cluster uniformity: {e}")
            raise
    
    def analyze_material_clusters(self, original_path: str, replacement_path: str) -> Dict[str, Any]:
        """
        Complete cluster-based material analysis
        
        Args:
            original_path: Path to original material image
            replacement_path: Path to replacement material image
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting cluster-based material analysis")
            
            # Load and preprocess images
            original_image = self.load_and_preprocess_image(original_path)
            replacement_image = self.load_and_preprocess_image(replacement_path)
            
            # Extract color clusters
            original_color_clusters = self.extract_color_clusters(original_image)
            replacement_color_clusters = self.extract_color_clusters(replacement_image)
            
            # Extract texture features
            original_texture = self.extract_texture_clusters(original_image)
            replacement_texture = self.extract_texture_clusters(replacement_image)
            
            # Calculate uniformity
            uniformity_results = self.calculate_cluster_uniformity(
                original_color_clusters, 
                replacement_color_clusters
            )
            
            # Combine results
            analysis_results = {
                'analysis_type': 'cluster_based_material_analysis',
                'image_dimensions': original_image.shape[:2],
                'n_clusters': self.n_clusters,
                'original_clusters': {
                    'color_analysis': original_color_clusters,
                    'texture_analysis': original_texture
                },
                'replacement_clusters': {
                    'color_analysis': replacement_color_clusters,
                    'texture_analysis': replacement_texture
                },
                'uniformity_assessment': uniformity_results,
                'quality_metrics': {
                    'original_silhouette_score': original_color_clusters['silhouette_score'],
                    'replacement_silhouette_score': replacement_color_clusters['silhouette_score']
                }
            }
            
            logger.info(f"Cluster analysis complete. Overall assessment: {uniformity_results['overall_assessment']}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in material cluster analysis: {e}")
            raise

def main():
    """Test the cluster analysis system"""
    analyzer = ClusterMaterialAnalyzer(n_clusters=4)
    
    # Example usage
    try:
        results = analyzer.analyze_material_clusters(
            'path/to/original.jpg',
            'path/to/replacement.jpg'
        )
        
        print("Cluster Analysis Results:")
        print(f"Overall Assessment: {results['uniformity_assessment']['overall_assessment']}")
        print(f"Average Delta-E: {results['uniformity_assessment']['average_delta_e']:.2f}")
        print(f"Confidence: {results['uniformity_assessment']['confidence_percentage']}%")
        
        for cluster in results['uniformity_assessment']['cluster_comparisons']:
            print(f"Cluster {cluster['cluster_id']}: ΔE={cluster['delta_e']:.2f}, Grade={cluster['uniformity_grade']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()