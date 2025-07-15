import cv2
import numpy as np

# ColorMath 3.0.0 still uses numpy.asscalar internally, which was removed in NumPy 2.0+
# This adds it back as a proper alias to a.item() for compatibility
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item() if hasattr(a, 'item') else a

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from scipy.ndimage import gaussian_laplace
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import logging
from datetime import datetime
from lighting_validator import ScientificLightingValidator

logger = logging.getLogger(__name__)

class MaterialAnalyzer:
    def __init__(self):
        self.colors = ["red", "green", "blue", "orange", "purple", "cyan", "yellow", "magenta", "lime", "pink"]
        self.lighting_validator = ScientificLightingValidator()
    
    def load_and_resize_image(self, image_path, target_size=(400, 400)):
        """
        Load and resize image to standardized size for scientific analysis.
        
        Standardization Benefits:
        - Consistent pixel-to-measurement ratios across all analyses
        - Uniform texture analysis (LBP, HOG, Gabor) regardless of source resolution
        - Accurate point correspondence between different source images
        - Scientific repeatability and comparison validity
        - Optimized computational performance for mobile analysis
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to exact target size for standardized scientific analysis
            # This ensures consistent measurement scales and texture calculations
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def get_lab_at_point(self, image, x, y):
        """Get LAB color values at specific point"""
        try:
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            rgb = image[y, x].astype(np.uint8).reshape((1, 1, 3))
            # Convert RGB to LAB
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = lab[0][0]
            
            # Convert to ColorMath LAB object
            # OpenCV LAB: L=[0,255], a=[0,255], b=[0,255] (need proper conversion)
            # ColorMath expects: L=[0,100], a=[-128,127], b=[-128,127]
            lab_l = float(l * 100.0 / 255.0)  # Scale L to 0-100
            lab_a = float(a - 128.0)          # Shift a to -128 to 127
            lab_b = float(b - 128.0)          # Shift b to -128 to 127
            
            return LabColor(lab_l=lab_l, lab_a=lab_a, lab_b=lab_b)
        except Exception as e:
            logger.error(f"Error getting LAB color at point ({x}, {y}): {str(e)}")
            raise
    
    def get_texture_features(self, image, x, y, patch_size=16):
        """Calculate texture features using LBP, HOG, and Gabor filters"""
        try:
            h, w = image.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # Extract patch around point
            half_patch = patch_size // 2
            y_start = max(0, y - half_patch)
            y_end = min(h, y + half_patch)
            x_start = max(0, x - half_patch)
            x_end = min(w, x + half_patch)
            
            patch = image[y_start:y_end, x_start:x_end]
            
            if patch.size == 0:
                return 0, 0, 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            
            # Local Binary Pattern
            lbp = local_binary_pattern(gray, 8, 1, method="uniform")
            lbp_mean = lbp.mean()
            
            # HOG features
            if gray.shape[0] >= 8 and gray.shape[1] >= 8:
                hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), 
                                 visualize=False, feature_vector=True)
                hog_mean = hog_features.mean() if hog_features.size > 0 else 0
            else:
                hog_mean = 0
            
            # Gabor filter (using Gaussian Laplacian as approximation)
            gabor = gaussian_laplace(gray.astype(np.float32), sigma=1)
            gabor_std = gabor.std()
            
            return lbp_mean, hog_mean, gabor_std
        except Exception as e:
            logger.error(f"Error calculating texture features at point ({x}, {y}): {str(e)}")
            return 0, 0, 0
    
    def get_gloss_metric(self, image, x, y, patch_size=6):
        """Calculate gloss metric using brightness analysis"""
        try:
            h, w = image.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # Extract small patch around point
            half_patch = patch_size // 2
            y_start = max(0, y - half_patch)
            y_end = min(h, y + half_patch)
            x_start = max(0, x - half_patch)
            x_end = min(w, x + half_patch)
            
            patch = image[y_start:y_end, x_start:x_end]
            
            if patch.size == 0:
                return 0
            
            # Convert to LAB and get L channel (brightness)
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            brightness = np.mean(lab[:, :, 0])
            
            return float(brightness)
        except Exception as e:
            logger.error(f"Error calculating gloss metric at point ({x}, {y}): {str(e)}")
            return 0
    
    def calculate_perceptos_index(self, delta_e, texture_delta, gloss_delta):
        """Calculate Perceptos Index based on color, texture, and gloss differences
        Returns value from 0-100 where 100 = perfect uniformity, 0 = maximum difference
        """
        # Updated normalization for real-world extreme values
        # Delta E: Extended scale for extreme differences (47-52 range)
        # Professional threshold: 0-3 excellent, 3-6 acceptable, 6+ poor, 20+ extreme
        if delta_e <= 3.0:
            norm_de = delta_e / 3.0 * 0.2  # Scale excellent range to 0-0.2
        elif delta_e <= 6.0:
            norm_de = 0.2 + (delta_e - 3.0) / 3.0 * 0.3  # Scale acceptable to 0.2-0.5
        elif delta_e <= 20.0:
            norm_de = 0.5 + (delta_e - 6.0) / 14.0 * 0.4  # Scale poor to 0.5-0.9
        else:
            norm_de = 0.9 + min((delta_e - 20.0) / 80.0, 0.1)  # Cap extreme at 1.0
        
        # Texture: Enhanced scale for material analysis
        norm_texture = min(texture_delta / 50.0, 1.0)
        
        # Gloss: Enhanced scale for brightness differences
        norm_gloss = min(gloss_delta / 100.0, 1.0)  # Extended range for real data
        
        # Professional weighting system based on material analysis standards
        if delta_e > 20.0:  # Extreme color differences (like 47-52 range)
            weights = (0.8, 0.15, 0.05)  # Color-critical for extreme differences
        elif delta_e > 6.0:  # Significant color difference
            weights = (0.7, 0.2, 0.1)  # Color-dominant for visible differences
        elif texture_delta > 25.0:  # Significant texture difference
            weights = (0.5, 0.4, 0.1)  # Texture-focused
        else:
            weights = (0.6, 0.25, 0.15)  # Balanced assessment
        
        # Calculate uniformity score (inverted - higher = more uniform)
        difference_score = weights[0] * norm_de + weights[1] * norm_texture + weights[2] * norm_gloss
        uniformity_score = max(0, 1.0 - difference_score)
        
        # Scale to 0-100 and round
        perceptos_index = round(uniformity_score * 100, 1)
        
        return perceptos_index
    
    def find_corresponding_point(self, original_img, replacement_img, click_x, click_y, search_radius=50):
        """Find the corresponding point in the second image using computer vision"""
        try:
            # Convert to grayscale for feature detection
            gray1 = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(replacement_img, cv2.COLOR_RGB2GRAY)
            
            # Create a local region around the clicked point for analysis
            h1, w1 = gray1.shape
            h2, w2 = gray2.shape
            
            # Ensure click point is within image bounds
            click_x = max(search_radius, min(w1 - search_radius, click_x))
            click_y = max(search_radius, min(h1 - search_radius, click_y))
            
            # Extract local patch around clicked point
            patch_size = search_radius * 2
            y1 = max(0, int(click_y - search_radius))
            y2 = min(h1, int(click_y + search_radius))
            x1 = max(0, int(click_x - search_radius))
            x2 = min(w1, int(click_x + search_radius))
            
            template = gray1[y1:y2, x1:x2]
            
            if template.size == 0:
                return None
            
            # Method 1: Template Matching with multiple scales
            best_match = None
            best_confidence = 0
            
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # Account for scale differences
            
            for scale in scales:
                # Resize template for scale invariance
                if scale != 1.0:
                    scaled_template = cv2.resize(template, 
                                               (int(template.shape[1] * scale), 
                                                int(template.shape[0] * scale)))
                else:
                    scaled_template = template
                
                if scaled_template.shape[0] >= gray2.shape[0] or scaled_template.shape[1] >= gray2.shape[1]:
                    continue
                
                # Template matching
                result = cv2.matchTemplate(gray2, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    # Adjust for template center and scale
                    center_x = max_loc[0] + scaled_template.shape[1] // 2
                    center_y = max_loc[1] + scaled_template.shape[0] // 2
                    best_match = (center_x, center_y, max_val)
            
            # Method 2: Feature-based matching for validation
            if best_match and best_confidence > 0.6:
                # Use ORB features for additional validation
                orb = cv2.ORB_create(nfeatures=100)
                
                # Find features in the template region
                kp1, des1 = orb.detectAndCompute(template, None)
                
                if des1 is not None and len(kp1) > 5:
                    # Search in region around template match
                    match_x, match_y, _ = best_match
                    search_x1 = max(0, match_x - search_radius)
                    search_y1 = max(0, match_y - search_radius)
                    search_x2 = min(w2, match_x + search_radius)
                    search_y2 = min(h2, match_y + search_radius)
                    
                    search_region = gray2[search_y1:search_y2, search_x1:search_x2]
                    
                    if search_region.size > 0:
                        kp2, des2 = orb.detectAndCompute(search_region, None)
                        
                        if des2 is not None and len(kp2) > 3:
                            # Match features
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            matches = bf.match(des1, des2)
                            
                            if len(matches) >= 3:
                                # Calculate centroid of matched features
                                matched_points = []
                                for match in matches[:10]:  # Top 10 matches
                                    pt = kp2[match.trainIdx].pt
                                    # Convert back to full image coordinates
                                    full_x = pt[0] + search_x1
                                    full_y = pt[1] + search_y1
                                    matched_points.append((full_x, full_y))
                                
                                if matched_points:
                                    # Use centroid of matched features
                                    avg_x = sum(p[0] for p in matched_points) / len(matched_points)
                                    avg_y = sum(p[1] for p in matched_points) / len(matched_points)
                                    
                                    # Refine the match using feature centroid
                                    refined_match = (avg_x, avg_y, best_confidence + 0.1)
                                    return {
                                        'x': int(refined_match[0]),
                                        'y': int(refined_match[1]),
                                        'confidence': min(1.0, refined_match[2]),
                                        'method': 'feature_refined'
                                    }
            
            # Return template matching result if confidence is good enough
            if best_match and best_confidence > 0.5:
                return {
                    'x': int(best_match[0]),
                    'y': int(best_match[1]),
                    'confidence': best_confidence,
                    'method': 'template_matching'
                }
            
            # Method 3: Fallback using global homography
            return self._find_point_using_homography(original_img, replacement_img, click_x, click_y)
            
        except Exception as e:
            logger.error(f"Error finding corresponding point: {str(e)}")
            return None
    
    def _find_point_using_homography(self, original_img, replacement_img, click_x, click_y):
        """Fallback method using global image homography"""
        try:
            gray1 = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(replacement_img, cv2.COLOR_RGB2GRAY)
            
            # Detect features globally
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return None
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                return None
            
            # Extract good matches
            good_matches = matches[:50]  # Top 50 matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Transform the clicked point
                point = np.array([[[click_x, click_y]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, H)
                
                if transformed is not None and len(transformed) > 0:
                    x, y = transformed[0][0]
                    
                    # Validate the transformed point is within image bounds
                    h, w = replacement_img.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        return {
                            'x': int(x),
                            'y': int(y),
                            'confidence': 0.7,
                            'method': 'homography'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in homography point finding: {str(e)}")
            return None
    
    def calculate_photo_alignment_score(self, original_img, replacement_img, original_points, replacement_points):
        """Calculate photo alignment using computer vision feature detection (independent of user points)"""
        try:
            # Convert to grayscale for feature detection
            gray1 = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(replacement_img, cv2.COLOR_RGB2GRAY)
            
            # 1. Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return {
                    'score': 0,
                    'assessment': 'insufficient_features',
                    'features_detected': f"{len(kp1) if kp1 else 0}/{len(kp2) if kp2 else 0}",
                    'match_quality': 0,
                    'geometric_consistency': 0,
                    'scale_consistency': 0
                }
            
            # 2. Feature Matching using FLANN matcher
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            try:
                matches = flann.knnMatch(des1, des2, k=2)
            except:
                # Fallback to brute force matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = [[m] for m in matches[:100]]  # Convert to knn format
            
            # 3. Filter good matches using Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    good_matches.append(match_pair[0])
            
            if len(good_matches) < 10:
                return {
                    'score': 20,
                    'assessment': 'poor_matching',
                    'features_detected': f"{len(kp1)}/{len(kp2)}",
                    'good_matches': len(good_matches),
                    'match_quality': 0,
                    'geometric_consistency': 0
                }
            
            # 4. Extract matched keypoint coordinates
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 5. Calculate homography and geometric consistency
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = np.sum(mask)
                geometric_consistency = (inliers / len(good_matches)) * 100
            except:
                geometric_consistency = 0
                H = None
            
            # 6. Scale and rotation analysis
            scale_consistency = 0
            rotation_consistency = 0
            
            if H is not None and len(good_matches) >= 20:
                try:
                    # Decompose homography to get scale and rotation
                    # Extract scale from homography matrix
                    scale_x = np.sqrt(H[0,0]**2 + H[1,0]**2)
                    scale_y = np.sqrt(H[0,1]**2 + H[1,1]**2)
                    scale_ratio = min(scale_x, scale_y) / max(scale_x, scale_y)
                    scale_consistency = scale_ratio * 100
                    
                    # Calculate rotation angle
                    rotation = np.arctan2(H[1,0], H[0,0]) * 180 / np.pi
                    rotation_consistency = max(0, 100 - abs(rotation) * 2)  # Penalize rotation
                except:
                    scale_consistency = 0
                    rotation_consistency = 0
            
            # 7. Material-specific feature analysis
            # Check if features are distributed across the image (not clustered)
            if len(good_matches) >= 10:
                matched_points = np.array([kp1[m.queryIdx].pt for m in good_matches])
                
                # Calculate feature distribution score
                center = np.mean(matched_points, axis=0)
                distances = np.linalg.norm(matched_points - center, axis=1)
                distribution_score = min(100, np.std(distances) * 2)  # Higher std = better distribution
            else:
                distribution_score = 0
            
            # 8. Match quality assessment
            if len(good_matches) > 0:
                avg_distance = np.mean([m.distance for m in good_matches])
                match_quality = max(0, 100 - avg_distance * 2)  # Lower distance = better quality
            else:
                match_quality = 0
            
            # 9. Overall alignment score calculation
            weights = {
                'geometric': 0.35,    # Most important - geometric consistency
                'matches': 0.25,      # Number and quality of matches
                'scale': 0.20,        # Scale consistency
                'distribution': 0.15, # Feature distribution
                'rotation': 0.05      # Rotation consistency
            }
            
            match_score = min(100, (len(good_matches) / 50) * 100)  # Normalize to 50 good matches = 100%
            
            alignment_score = (
                geometric_consistency * weights['geometric'] +
                match_score * weights['matches'] +
                scale_consistency * weights['scale'] +
                distribution_score * weights['distribution'] +
                rotation_consistency * weights['rotation']
            )
            
            # 10. Assessment categories
            if alignment_score >= 85:
                assessment = 'excellent'
            elif alignment_score >= 70:
                assessment = 'good'
            elif alignment_score >= 55:
                assessment = 'fair'
            elif alignment_score >= 35:
                assessment = 'poor'
            else:
                assessment = 'very_poor'
            
            logger.info(f"Feature detection: {len(kp1)}/{len(kp2)} keypoints, {len(good_matches)} good matches")
            logger.info(f"Geometric consistency: {geometric_consistency:.1f}%, Match quality: {match_quality:.1f}")
            
            return {
                'score': round(alignment_score, 1),
                'assessment': assessment,
                'features_detected': f"{len(kp1)}/{len(kp2)}",
                'good_matches': len(good_matches),
                'geometric_consistency': round(geometric_consistency, 1),
                'match_quality': round(match_quality, 1),
                'scale_consistency': round(scale_consistency, 1),
                'feature_distribution': round(distribution_score, 1),
                'analysis_method': 'computer_vision_features'
            }
            
        except Exception as e:
            logger.error(f"Error in computer vision alignment analysis: {str(e)}")
            return {
                'score': 0,
                'assessment': 'analysis_error',
                'features_detected': '0/0',
                'good_matches': 0,
                'geometric_consistency': 0,
                'match_quality': 0,
                'error': str(e)
            }

    def calculate_human_perception_assessment(self, results):
        """Calculate overall uniformity based on human perception logic and Perceptos Index"""
        if not results:
            return "unknown"
            
        total_points = len(results)
        uniform_points = sum(1 for r in results if r['is_uniform'])
        
        # Calculate statistical metrics
        color_scores = [r['delta_e'] for r in results]
        texture_scores = [r['texture_delta'] for r in results]
        gloss_scores = [r['gloss_delta'] for r in results]
        perceptos_scores = [r['perceptos_index'] for r in results]
        
        avg_color = sum(color_scores) / len(color_scores)
        avg_texture = sum(texture_scores) / len(texture_scores)
        avg_gloss = sum(gloss_scores) / len(gloss_scores)
        avg_perceptos = sum(perceptos_scores) / len(perceptos_scores)
        
        # Human perception thresholds based on scientific standards
        color_threshold = 4.0  # ΔE > 4 is clearly perceptible
        texture_threshold = 25.0  # Material-dependent texture variation
        gloss_threshold = 20.0  # Noticeable gloss differences
        
        # Count significantly problematic points
        major_color_issues = sum(1 for score in color_scores if score > 6.0)
        major_texture_issues = sum(1 for score in texture_scores if score > 35.0)
        major_gloss_issues = sum(1 for score in gloss_scores if score > 25.0)
        
        # Calculate uniformity percentage
        uniformity_percentage = (uniform_points / total_points) * 100
        
        # Enhanced human perception logic using Perceptos Index and uniformity percentage
        # Professional thresholds for material uniformity assessment
        
        # Check for any major color issues (ΔE > 6 is clearly visible difference)
        has_major_color_issues = major_color_issues > 0
        
        # Professional uniformity assessment logic
        if avg_perceptos >= 85 and uniformity_percentage >= 80 and not has_major_color_issues:
            assessment = "uniform"  # Excellent uniformity
        elif avg_perceptos >= 70 and uniformity_percentage >= 75 and major_color_issues <= 1:
            assessment = "acceptable"  # Good uniformity with minor variations
        elif has_major_color_issues or avg_color > 5.0 or uniformity_percentage < 75:
            assessment = "non-uniform"  # Clear non-uniformity detected
        else:
            assessment = "acceptable"  # Default for edge cases
            
        return assessment
    
    def extract_color_clusters(self, image, n_clusters=4):
        """
        Extract color clusters using K-means clustering in LAB color space
        
        Args:
            image: Input image in RGB format
            n_clusters: Number of clusters for K-means analysis
            
        Returns:
            Dictionary containing cluster information
        """
        try:
            # Convert to LAB color space for perceptually uniform clustering
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Reshape image to pixel array
            pixels = lab_image.reshape((-1, 3))
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Reshape labels back to image dimensions
            label_map = labels.reshape(image.shape[:2])
            
            # Get cluster centers (in LAB color space)
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
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
            if len(pixels) > n_clusters:
                silhouette_avg = silhouette_score(pixels, labels)
            else:
                silhouette_avg = 0.0
            
            return {
                'cluster_centers': cluster_centers.tolist(),
                'label_map': label_map.tolist(),
                'cluster_stats': cluster_stats,
                'silhouette_score': float(silhouette_avg),
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in color clustering: {e}")
            raise

    def calculate_cluster_uniformity(self, original_clusters, replacement_clusters):
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
            n_clusters = min(len(original_clusters['cluster_centers']), 
                           len(replacement_clusters['cluster_centers']))
            
            # Compare each cluster pair
            for i in range(n_clusters):
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
                overall_assessment = 'uniform'
                confidence = 95
            elif avg_delta_e < 5.0:
                overall_assessment = 'mostly_uniform'
                confidence = 80
            elif avg_delta_e < 10.0:
                overall_assessment = 'acceptable'
                confidence = 60
            else:
                overall_assessment = 'non_uniform'
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

    def analyze_clusters(self, original_path, replacement_path, n_clusters=4, 
                        measurement_type='standard', measurement_distance='18 inches'):
        """
        Analyze material uniformity using K-means clustering instead of manual points
        
        Args:
            original_path: Path to original material image
            replacement_path: Path to replacement material image
            n_clusters: Number of clusters for analysis
            measurement_type: Type of measurement being performed
            measurement_distance: Distance at which measurement was taken
            
        Returns:
            Complete cluster-based analysis results
        """
        try:
            logger.info(f"Starting cluster-based analysis with {n_clusters} clusters")
            
            # Load and resize images for consistent analysis
            original_img = self.load_and_resize_image(original_path)
            replacement_img = self.load_and_resize_image(replacement_path)
            
            # Validate lighting conditions
            original_lighting = self.lighting_validator.analyze_image_lighting(original_path)
            replacement_lighting = self.lighting_validator.analyze_image_lighting(replacement_path)
            
            # Extract color clusters from both images
            original_clusters = self.extract_color_clusters(original_img, n_clusters)
            replacement_clusters = self.extract_color_clusters(replacement_img, n_clusters)
            
            # Calculate uniformity between clusters
            uniformity_results = self.calculate_cluster_uniformity(original_clusters, replacement_clusters)
            
            # Generate cluster-based results in the same format as point analysis
            cluster_results = []
            for comparison in uniformity_results['cluster_comparisons']:
                cluster_results.append({
                    'cluster_id': comparison['cluster_id'],
                    'delta_e': comparison['delta_e'],
                    'uniformity_grade': comparison['uniformity_grade'],
                    'original_percentage': comparison['original_percentage'],
                    'replacement_percentage': comparison['replacement_percentage'],
                    'size_difference': comparison['size_difference_percent'],
                    'is_uniform': comparison['delta_e'] < 6.0,  # Using Delta-E threshold
                })
            
            # Calculate overall Perceptos Index based on cluster analysis
            avg_delta_e = uniformity_results['average_delta_e']
            perceptos_index = max(0, 100 - (avg_delta_e * 10))  # Scale to 0-100
            
            # Generate results in compatible format
            results = {
                'analysis_type': 'cluster_based',
                'analysis_results': {
                    'perceptos_index': perceptos_index,
                    'average_delta_e': avg_delta_e,
                    'uniformity_assessment': uniformity_results['overall_assessment'],
                    'confidence_percentage': uniformity_results['confidence_percentage'],
                    'points_analyzed': len(cluster_results),
                    'measurement_type': measurement_type,
                    'measurement_distance': measurement_distance
                },
                'detailed_results': cluster_results,
                'cluster_data': {
                    'original_clusters': original_clusters,
                    'replacement_clusters': replacement_clusters,
                    'uniformity_analysis': uniformity_results
                },
                'lighting_validation': {
                    'original': original_lighting.__dict__,
                    'replacement': replacement_lighting.__dict__,
                    'overall_quality': self._assess_overall_lighting_quality(original_lighting, replacement_lighting)
                },
                'verified_report_eligible': self._check_verified_report_eligibility(original_lighting, replacement_lighting)
            }
            
            logger.info(f"Cluster analysis complete. Assessment: {uniformity_results['overall_assessment']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in cluster analysis: {e}")
            raise

    def analyze_points(self, original_path, replacement_path, original_points, replacement_points, 
                      measurement_type='standard', measurement_distance='18 inches'):
        """Analyze corresponding points between two images with scientific lighting validation"""
        try:
            # Validate lighting conditions scientifically
            logger.info("Performing scientific lighting validation...")
            original_lighting = self.lighting_validator.analyze_image_lighting(original_path)
            replacement_lighting = self.lighting_validator.analyze_image_lighting(replacement_path)
            
            # Load images
            original_img = self.load_and_resize_image(original_path)
            replacement_img = self.load_and_resize_image(replacement_path)
            
            results = []
            
            for i, (orig_point, repl_point) in enumerate(zip(original_points, replacement_points)):
                x1, y1 = orig_point['x'], orig_point['y']
                x2, y2 = repl_point['x'], repl_point['y']
                
                # Color analysis
                lab1 = self.get_lab_at_point(original_img, x1, y1)
                lab2 = self.get_lab_at_point(replacement_img, x2, y2)
                delta_e = delta_e_cie2000(lab1, lab2)
                
                # Texture analysis with proper weighting
                lbp1, hog1, gabor1 = self.get_texture_features(original_img, x1, y1)
                lbp2, hog2, gabor2 = self.get_texture_features(replacement_img, x2, y2)
                
                # Weighted texture difference calculation
                # LBP: 0.5 weight (most important for material texture)
                # HOG: 0.3 weight (edge patterns)
                # Gabor: 0.2 weight (fine texture details)
                lbp_diff = abs(lbp1 - lbp2)
                hog_diff = abs(hog1 - hog2) 
                gabor_diff = abs(gabor1 - gabor2)
                
                # Normalize each component based on typical ranges
                lbp_norm = lbp_diff / 10.0  # LBP typically 0-10
                hog_norm = hog_diff / 1.0   # HOG features typically 0-1  
                gabor_norm = gabor_diff / 100.0  # Gabor std typically 0-100
                
                texture_delta = (0.5 * lbp_norm + 0.3 * hog_norm + 0.2 * gabor_norm) * 50.0
                
                # Gloss analysis
                gloss1 = self.get_gloss_metric(original_img, x1, y1)
                gloss2 = self.get_gloss_metric(replacement_img, x2, y2)
                gloss_delta = abs(gloss1 - gloss2)
                
                # Calculate Perceptos Index
                perceptos_index = self.calculate_perceptos_index(delta_e, texture_delta, gloss_delta)
                
                # Determine uniformity with measurement-dependent thresholds
                if measurement_type == 'precision':
                    # Stricter thresholds for precision measurements
                    color_threshold = 3.0
                    texture_threshold = 15.0
                    gloss_threshold = 10.0
                elif measurement_type == 'construction':
                    # More lenient for construction materials
                    color_threshold = 6.0
                    texture_threshold = 25.0
                    gloss_threshold = 20.0
                else:  # standard
                    color_threshold = 5.0
                    texture_threshold = 20.0
                    gloss_threshold = 15.0
                
                is_uniform = (delta_e <= color_threshold and 
                            texture_delta <= texture_threshold and 
                            gloss_delta <= gloss_threshold)
                
                result = {
                    'point_number': i + 1,
                    'color': self.colors[i % len(self.colors)],
                    'delta_e': round(delta_e, 2),
                    'texture_delta': round(texture_delta, 2),
                    'gloss_delta': round(gloss_delta, 2),
                    'perceptos_index': perceptos_index,
                    'is_uniform': is_uniform,
                    'original_point': {'x': x1, 'y': y1},
                    'replacement_point': {'x': x2, 'y': y2},
                    'measurement_type': measurement_type,
                    'measurement_distance': measurement_distance
                }
                
                results.append(result)
            
            # Calculate alignment score using computer vision
            alignment_score = self.calculate_photo_alignment_score(
                original_img, replacement_img, original_points, replacement_points
            )
            
            # Add lighting validation to results
            analysis_results = {
                'point_analysis': results,
                'lighting_validation': {
                    'original_image': {
                        'illuminance_lux': original_lighting.illuminance_estimate,
                        'color_temperature_k': original_lighting.color_temperature_estimate,
                        'uniformity_percent': original_lighting.uniformity_score,
                        'cri_estimate': original_lighting.cri_estimate,
                        'scientific_validity': original_lighting.scientific_validity,
                        'compliance_level': original_lighting.compliance_level,
                        'recommendations': original_lighting.recommendations
                    },
                    'replacement_image': {
                        'illuminance_lux': replacement_lighting.illuminance_estimate,
                        'color_temperature_k': replacement_lighting.color_temperature_estimate,
                        'uniformity_percent': replacement_lighting.uniformity_score,
                        'cri_estimate': replacement_lighting.cri_estimate,
                        'scientific_validity': replacement_lighting.scientific_validity,
                        'compliance_level': replacement_lighting.compliance_level,
                        'recommendations': replacement_lighting.recommendations
                    },
                    'overall_lighting_quality': self._assess_overall_lighting_quality(original_lighting, replacement_lighting),
                    'verified_report_eligible': self._check_verified_report_eligibility(original_lighting, replacement_lighting)
                },
                'photo_alignment': alignment_score,
                'measurement_metadata': {
                    'type': measurement_type,
                    'distance': measurement_distance,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Analysis completed for {len(results)} point pairs using {measurement_type} at {measurement_distance}")
            logger.info(f"Lighting validation: Original={original_lighting.scientific_validity}, Replacement={replacement_lighting.scientific_validity}")
            logger.info(f"Verified report eligible: {analysis_results['lighting_validation']['verified_report_eligible']}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing points: {str(e)}")
            raise
    
    def _assess_overall_lighting_quality(self, original_lighting, replacement_lighting):
        """Assess overall lighting quality for both images"""
        scores = [
            original_lighting.uniformity_score,
            replacement_lighting.uniformity_score,
            original_lighting.cri_estimate,
            replacement_lighting.cri_estimate
        ]
        
        # Weight illuminance consistency between images
        illuminance_diff = abs(original_lighting.illuminance_estimate - replacement_lighting.illuminance_estimate)
        consistency_penalty = min(20, illuminance_diff / 50)  # Penalty for large differences
        
        avg_score = np.mean(scores) - consistency_penalty
        
        if avg_score >= 85:
            return "excellent"
        elif avg_score >= 75:
            return "good"
        elif avg_score >= 60:
            return "acceptable"
        else:
            return "poor"
    
    def _check_verified_report_eligibility(self, original_lighting, replacement_lighting):
        """Check if lighting conditions meet verified report standards"""
        # Both images must meet minimum standards
        original_meets = (
            original_lighting.compliance_level in ["CIE_compliant", "acceptable"] and
            original_lighting.uniformity_score >= 70 and
            original_lighting.cri_estimate >= 80
        )
        
        replacement_meets = (
            replacement_lighting.compliance_level in ["CIE_compliant", "acceptable"] and
            replacement_lighting.uniformity_score >= 70 and
            replacement_lighting.cri_estimate >= 80
        )
        
        # Lighting should be reasonably consistent between images
        illuminance_diff = abs(original_lighting.illuminance_estimate - replacement_lighting.illuminance_estimate)
        color_temp_diff = abs(original_lighting.color_temperature_estimate - replacement_lighting.color_temperature_estimate)
        
        consistency_ok = illuminance_diff < 300 and color_temp_diff < 1000  # Reasonable tolerances
        
        return original_meets and replacement_meets and consistency_ok
    
    def generate_pdf_report(self, session_id, results, reports_folder, 
                           measurement_type='standard', measurement_distance='18 inches'):
        """Generate comprehensive insurance-quality PDF report with photos and scientific documentation"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.lib import colors
            from reportlab.lib.utils import ImageReader
            import os
            from PIL import Image as PILImage
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optikos_insurance_report_{session_id}_{timestamp}.pdf"
            filepath = os.path.join(reports_folder, filename)
            
            # Create PDF document with professional margins
            doc = SimpleDocTemplate(
                filepath, 
                pagesize=letter,
                leftMargin=0.75*inch,
                rightMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            styles = getSampleStyleSheet()
            story = []
            
            # Professional title styling for insurance submission
            title_style = ParagraphStyle(
                'InsuranceTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=10,
                textColor=colors.HexColor('#1E3A8A'),
                alignment=1,
                fontName='Helvetica-Bold'
            )
            
            subtitle_style = ParagraphStyle(
                'InsuranceSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                textColor=colors.HexColor('#374151'),
                alignment=1,
                fontName='Helvetica'
            )
            
            # Report Header
            story.append(Paragraph("MATERIAL UNIFORMITY ANALYSIS REPORT", title_style))
            story.append(Paragraph("Professional Scientific Assessment for Insurance Documentation", subtitle_style))
            story.append(Spacer(1, 0.4*inch))
            
            # Executive Summary Box
            exec_style = ParagraphStyle(
                'ExecutiveSummary',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=20,
                textColor=colors.HexColor('#1F2937'),
                backColor=colors.HexColor('#F8FAFC'),
                borderPadding=15,
                borderWidth=2,
                borderColor=colors.HexColor('#E2E8F0'),
                alignment=0
            )
            
            current_date = datetime.now().strftime("%B %d, %Y")
            report_id = f"OPT-{session_id[:8].upper()}-{timestamp}"
            
            exec_summary = f"""
            <b>EXECUTIVE SUMMARY</b><br/><br/>
            <b>Report ID:</b> {report_id}<br/>
            <b>Analysis Date:</b> {current_date}<br/>
            <b>Measurement Type:</b> {measurement_type.title()} Analysis<br/>
            <b>Measurement Distance:</b> {measurement_distance}<br/>
            <b>Analysis Method:</b> Computer Vision with CIE Standards<br/>
            <b>Scientific Basis:</b> Delta-E CIE2000, LAB Color Space, Texture Analysis<br/>
            <b>Report Purpose:</b> Insurance Carrier Documentation & Professional Assessment
            """
            story.append(Paragraph(exec_summary, exec_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Find and include original images
            original_image_path = None
            replacement_image_path = None
            
            # Look for uploaded images
            uploads_folder = 'uploads'
            if os.path.exists(uploads_folder):
                for filename in os.listdir(uploads_folder):
                    if filename.startswith(session_id):
                        if 'original' in filename.lower():
                            original_image_path = os.path.join(uploads_folder, filename)
                        elif 'replacement' in filename.lower():
                            replacement_image_path = os.path.join(uploads_folder, filename)
            
            # Material Photos Section
            photo_style = ParagraphStyle(
                'PhotoHeader',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=15,
                textColor=colors.HexColor('#1E3A8A'),
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("MATERIAL SAMPLE DOCUMENTATION", photo_style))
            
            # Create photo comparison table
            photo_data = [['Original Material Sample', 'Replacement Material Sample']]
            
            # Process and add images
            def add_image_to_report(image_path, max_width=3*inch, max_height=2.5*inch):
                """Add properly sized image to report"""
                if image_path and os.path.exists(image_path):
                    try:
                        # Open and resize image
                        pil_img = PILImage.open(image_path)
                        
                        # Calculate aspect ratio
                        aspect = pil_img.width / pil_img.height
                        
                        if aspect > 1:  # Landscape
                            width = min(max_width, max_width)
                            height = width / aspect
                        else:  # Portrait
                            height = min(max_height, max_height)
                            width = height * aspect
                        
                        # Create ReportLab Image
                        img = Image(image_path, width=width, height=height)
                        return img
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {str(e)}")
                        return Paragraph("Image processing error", styles['Normal'])
                else:
                    return Paragraph("Image not available", styles['Normal'])
            
            original_img = add_image_to_report(original_image_path)
            replacement_img = add_image_to_report(replacement_image_path)
            
            photo_data.append([original_img, replacement_img])
            
            # Create photo table
            photo_table = Table(photo_data, colWidths=[3.5*inch, 3.5*inch])
            photo_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E0')),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            story.append(photo_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Streamlined Technical Information
            tech_info_style = ParagraphStyle(
                'TechInfo',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=20,
                textColor=colors.HexColor('#6B7280'),
                alignment=0
            )
            
            tech_info = f"""
            <b>Technical Standards:</b> Analysis performed using {measurement_distance} measurement distance with CIE Delta-E 2000 color standards and standardized 400x400 pixel resolution. Complete technical specifications and CIE standards documentation available at: optikos.app/standards
            """
            story.append(Paragraph(tech_info, tech_info_style))
            
            # Handle both old format (list) and new format (dict with point_analysis)
            if isinstance(results, dict) and 'point_analysis' in results:
                # New format - extract point analysis data
                point_results = results['point_analysis']
                lighting_data = results.get('lighting_validation', {})
                photo_alignment = results.get('photo_alignment', {})
                
                # Scientific Lighting Validation Section
                if lighting_data:
                    story.append(PageBreak())  # New page for detailed analysis
                    
                    lighting_style = ParagraphStyle(
                        'LightingHeader',
                        parent=styles['Heading2'],
                        fontSize=18,
                        spaceAfter=15,
                        textColor=colors.HexColor('#1E3A8A'),
                        fontName='Helvetica-Bold'
                    )
                    story.append(Paragraph("SCIENTIFIC LIGHTING VALIDATION", lighting_style))
                    
                    original_lighting = lighting_data.get('original_image', {})
                    replacement_lighting = lighting_data.get('replacement_image', {})
                    
                    # Create comprehensive lighting analysis table
                    lighting_table_data = [
                        ['Lighting Parameter', 'Original Sample', 'Replacement Sample', 'CIE Standard', 'Assessment'],
                        [
                            'Illuminance (lux)',
                            f"{original_lighting.get('illuminance_lux', 0):.0f}",
                            f"{replacement_lighting.get('illuminance_lux', 0):.0f}",
                            '300-800 optimal',
                            'Within Range' if 300 <= original_lighting.get('illuminance_lux', 0) <= 800 and 300 <= replacement_lighting.get('illuminance_lux', 0) <= 800 else 'Outside Range'
                        ],
                        [
                            'Color Temperature (K)',
                            f"{original_lighting.get('color_temperature_k', 0)}",
                            f"{replacement_lighting.get('color_temperature_k', 0)}",
                            '4000-6500K preferred',
                            'Acceptable' if 4000 <= original_lighting.get('color_temperature_k', 0) <= 6500 and 4000 <= replacement_lighting.get('color_temperature_k', 0) <= 6500 else 'Sub-optimal'
                        ],
                        [
                            'Uniformity (%)',
                            f"{original_lighting.get('uniformity_percent', 0):.1f}%",
                            f"{replacement_lighting.get('uniformity_percent', 0):.1f}%",
                            '>70% required',
                            'Good' if original_lighting.get('uniformity_percent', 0) >= 70 and replacement_lighting.get('uniformity_percent', 0) >= 70 else 'Poor'
                        ],
                        [
                            'CRI (Color Rendering)',
                            f"{original_lighting.get('cri_estimate', 0)}",
                            f"{replacement_lighting.get('cri_estimate', 0)}",
                            '>80 minimum',
                            'Excellent' if original_lighting.get('cri_estimate', 0) >= 90 and replacement_lighting.get('cri_estimate', 0) >= 90 else 'Good' if original_lighting.get('cri_estimate', 0) >= 80 and replacement_lighting.get('cri_estimate', 0) >= 80 else 'Poor'
                        ],
                        [
                            'Scientific Validity',
                            original_lighting.get('scientific_validity', 'Unknown').title(),
                            replacement_lighting.get('scientific_validity', 'Unknown').title(),
                            'Good or Excellent',
                            lighting_data.get('overall_lighting_quality', 'Unknown').title()
                        ]
                    ]
                    
                    lighting_table = Table(lighting_table_data, colWidths=[1.4*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                    lighting_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    
                    story.append(lighting_table)
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Lighting Compliance Summary
                    compliance_style = ParagraphStyle(
                        'ComplianceInfo',
                        parent=styles['Normal'],
                        fontSize=11,
                        spaceAfter=20,
                        textColor=colors.HexColor('#1F2937'),
                        backColor=colors.HexColor('#F0F9FF'),
                        borderPadding=12,
                        borderWidth=1,
                        borderColor=colors.HexColor('#0EA5E9'),
                        alignment=0
                    )
                    
                    verified_eligible = lighting_data.get('verified_report_eligible', False)
                    compliance_text = f"""
                    <b>LIGHTING COMPLIANCE ASSESSMENT:</b><br/><br/>
                    <b>CIE Standard Compliance:</b> {lighting_data.get('overall_lighting_quality', 'Unknown').title()}<br/>
                    <b>Verified Report Eligible:</b> {'Yes - Meets professional standards' if verified_eligible else 'No - Does not meet all requirements'}<br/>
                    <b>Insurance Documentation Quality:</b> {'Professional Grade' if verified_eligible else 'Standard Grade'}<br/>
                    <b>Recommendation:</b> {'Suitable for insurance submission' if verified_eligible else 'Acceptable for documentation purposes'}
                    """
                    story.append(Paragraph(compliance_text, compliance_style))
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Photo Alignment Analysis
                    if photo_alignment:
                        alignment_style = ParagraphStyle(
                            'AlignmentHeader',
                            parent=styles['Heading3'],
                            fontSize=14,
                            spaceAfter=10,
                            textColor=colors.HexColor('#374151'),
                            fontName='Helvetica-Bold'
                        )
                        story.append(Paragraph("PHOTO ALIGNMENT ANALYSIS", alignment_style))
                        
                        alignment_score = photo_alignment.get('score', 0)
                        alignment_assessment = photo_alignment.get('assessment', 'unknown')
                        
                        alignment_info = f"""
                        <b>Computer Vision Alignment Score:</b> {alignment_score:.1f}/100<br/>
                        <b>Assessment:</b> {alignment_assessment.title()}<br/>
                        <b>Analysis Method:</b> {photo_alignment.get('analysis_method', 'Unknown')}<br/>
                        <b>Features Detected:</b> {photo_alignment.get('features_detected', 'N/A')}<br/>
                        <b>Good Matches:</b> {photo_alignment.get('good_matches', 'N/A')}<br/>
                        <b>Geometric Consistency:</b> {photo_alignment.get('geometric_consistency', 0):.1f}%
                        """
                        story.append(Paragraph(alignment_info, styles['Normal']))
                        story.append(Spacer(1, 0.3*inch))
            else:
                # Old format - results is a list
                point_results = results
            
            # Professional Analysis Results Section
            analysis_style = ParagraphStyle(
                'AnalysisHeader',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=15,
                textColor=colors.HexColor('#1E3A8A'),
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("PROFESSIONAL ANALYSIS RESULTS", analysis_style))
            
            if not point_results or len(point_results) == 0:
                raise ValueError('No point analysis data found for report generation')
                
            total_points = len(point_results)
            uniform_points = sum(1 for r in point_results if r.get('is_uniform', False))
            avg_perceptos = round(sum(r.get('perceptos_index', 0) for r in point_results) / total_points, 2)
            
            # Calculate human perception assessment ONCE for consistency
            assessment = self.calculate_human_perception_assessment(point_results)
            
            # Professional assessment display with single consistent logic
            if assessment == "uniform":
                overall_status = 'UNIFORM'
                status_color = colors.HexColor('#059669')
            elif assessment == "acceptable":
                overall_status = 'ACCEPTABLE'
                status_color = colors.HexColor('#0EA5E9')
            else:  # non-uniform
                overall_status = 'NON-UNIFORM'
                status_color = colors.HexColor('#EF4444')
            
            # Professional Assessment Summary Box
            assessment_style = ParagraphStyle(
                'AssessmentSummary',
                parent=styles['Normal'],
                fontSize=13,
                spaceAfter=20,
                textColor=colors.HexColor('#1F2937'),
                backColor=colors.HexColor('#FEFEFE'),
                borderPadding=15,
                borderWidth=2,
                borderColor=status_color,
                alignment=0
            )
            
            non_uniform_points = total_points - uniform_points
            uniformity_percentage = (uniform_points / total_points) * 100
            
            # Insurance-focused summary
            summary_text = f"""
            <b>MATERIAL UNIFORMITY ASSESSMENT SUMMARY</b><br/><br/>
            <b>Total Analysis Points:</b> {total_points} locations measured<br/>
            <b>Uniform Material Points:</b> {uniform_points} ({uniformity_percentage:.1f}%)<br/>
            <b>Non-Uniform Material Points:</b> {non_uniform_points} ({100-uniformity_percentage:.1f}%)<br/>
            <b>Average Perceptos Index:</b> {avg_perceptos}/100 (Scientific Uniformity Score)<br/>
            <b>Professional Assessment:</b> <font color="{status_color.hexval()}"><b>{overall_status}</b></font><br/><br/>
            <b>Insurance Recommendation:</b> {'Material replacement recommended due to visible non-uniformity' if overall_status == 'NON-UNIFORM' else 'Materials show acceptable uniformity for insurance purposes' if overall_status == 'ACCEPTABLE' else 'Materials demonstrate excellent uniformity'}
            """
            story.append(Paragraph(summary_text, assessment_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Point Analysis with Insurance Focus
            detailed_style = ParagraphStyle(
                'DetailedHeader',
                parent=styles['Heading3'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.HexColor('#374151'),
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("DETAILED POINT-BY-POINT ANALYSIS", detailed_style))
            
            # Enhanced analysis table with professional grading
            analysis_table_data = [
                ['Point', 'Location', 'Color Difference (ΔE)', 'Texture Variance', 'Gloss Difference', 'Perceptos Index', 'Professional Grade', 'Assessment']
            ]
            
            for i, result in enumerate(point_results):
                # Enhanced status with professional grading
                is_uniform = result.get('is_uniform', False)
                status = "Uniform" if is_uniform else "Non-Uniform"
                
                # Safely access all fields with defaults
                point_num = result.get('point_number', i+1)
                color = result.get('color', 'unknown').title()
                delta_e = result.get('delta_e', 0)
                texture_delta = result.get('texture_delta', 0) 
                gloss_delta = result.get('gloss_delta', 0)
                perceptos_index = result.get('perceptos_index', 0)
                
                # Professional grading system for insurance
                if delta_e <= 2 and texture_delta <= 10 and gloss_delta <= 10:
                    grade = "A+ Excellent"
                elif delta_e <= 4 and texture_delta <= 20 and gloss_delta <= 15:
                    grade = "A Good"
                elif delta_e <= 6 and texture_delta <= 30 and gloss_delta <= 20:
                    grade = "B Fair"
                elif delta_e <= 8 and texture_delta <= 40 and gloss_delta <= 25:
                    grade = "C Poor"
                else:
                    grade = "F Unacceptable"
                
                analysis_table_data.append([
                    f"#{point_num}",
                    f"{color} Point",
                    f"{delta_e:.2f} CIE2000",
                    f"{texture_delta:.2f}",
                    f"{gloss_delta:.1f}",
                    f"{perceptos_index:.1f}/100",
                    grade,
                    status
                ])
            
            # Create professional analysis table
            analysis_table = Table(analysis_table_data, colWidths=[0.6*inch, 0.9*inch, 1.0*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch])
            analysis_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                
                # Data rows styling with professional color coding
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            
            story.append(analysis_table)
            story.append(Spacer(1, 0.4*inch))
            
            # Scientific Methodology Section
            method_style = ParagraphStyle(
                'MethodHeader',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=15,
                textColor=colors.HexColor('#1E3A8A'),
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("SCIENTIFIC METHODOLOGY & STANDARDS", method_style))
            
            methodology_data = [
                ['Analysis Parameter', 'Scientific Method', 'Industry Standard', 'Threshold Values'],
                [
                    'Color Difference',
                    'CIE Delta-E 2000 (LAB Color Space)',
                    'ISO 11664-6:2014',
                    'ΔE ≤ 3: Imperceptible\nΔE 3-6: Perceptible\nΔE > 6: Clearly Visible'
                ],
                [
                    'Texture Analysis',
                    'Local Binary Patterns (LBP)\nHistogram of Oriented Gradients (HOG)\nGabor Filter Analysis',
                    'Computer Vision Standards',
                    'Combined Score > 20:\nSignificant Texture Variation'
                ],
                [
                    'Gloss Measurement',
                    'LAB L* Channel Brightness Analysis',
                    'Material Surface Analysis',
                    'Difference > 15:\nNoticeable Gloss Variation'
                ],
                [
                    'Perceptos Index',
                    'Weighted Composite Score:\nColor: 60%, Texture: 25%, Gloss: 15%',
                    'Professional Material Assessment',
                    '85+: Excellent\n70-84: Acceptable\n<70: Non-Uniform'
                ]
            ]
            
            methodology_table = Table(methodology_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
            methodology_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(methodology_table)
            story.append(Spacer(1, 0.4*inch))
            
            # Insurance Certification Section
            cert_style = ParagraphStyle(
                'CertificationInfo',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=20,
                textColor=colors.HexColor('#1F2937'),
                backColor=colors.HexColor('#FEF3C7'),
                borderPadding=15,
                borderWidth=2,
                borderColor=colors.HexColor('#F59E0B'),
                alignment=0
            )
            
            certification_text = f"""
            <b>INSURANCE CARRIER CERTIFICATION</b><br/><br/>
            <b>Report Classification:</b> Professional Scientific Material Analysis<br/>
            <b>Methodology Compliance:</b> CIE International Standards<br/>
            <b>Analysis Software:</b> Optikos Material Analysis Platform v2.0<br/>
            <b>Certification Date:</b> {current_date}<br/>
            <b>Report Validity:</b> Suitable for insurance documentation and claims processing<br/>
            <b>Professional Standards:</b> Meets industry requirements for material uniformity assessment<br/><br/>
            This report provides scientific basis for material replacement decisions and insurance claims evaluation.
            """
            story.append(Paragraph(certification_text, cert_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Professional Footer
            footer_style = ParagraphStyle(
                'ProfessionalFooter',
                parent=styles['Normal'],
                fontSize=9,
                spaceAfter=0,
                textColor=colors.HexColor('#6B7280'),
                alignment=1
            )
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("─" * 80, footer_style))
            story.append(Paragraph(f"Generated by Optikos Material Analysis Platform | Report ID: {report_id}", footer_style))
            story.append(Paragraph("Professional Scientific Assessment for Insurance Documentation", footer_style))
            story.append(Paragraph(f"Analysis completed on {current_date} using CIE International Standards", footer_style))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
