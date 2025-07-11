import cv2
import numpy as np

# ColorMath 3.0.0 still uses numpy.asscalar internally, which was removed in NumPy 2.0+
# This adds it back as a proper alias to a.item() for compatibility
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item() if hasattr(a, 'item') else a

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from skimage.feature import local_binary_pattern, hog
from scipy.ndimage import gaussian_laplace
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MaterialAnalyzer:
    def __init__(self):
        self.colors = ["red", "green", "blue", "orange", "purple", "cyan", "yellow", "magenta", "lime", "pink"]
    
    def load_and_resize_image(self, image_path, target_size=(400, 400)):
        """Load and resize image to exact same size for accurate point correspondence"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to exact target size for precise point correspondence
            # This ensures both images are exactly the same dimensions
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
        # Scientific normalization based on human perception thresholds
        # Delta E: 0-10 scale (> 5 is noticeable, > 10 is significant)
        norm_de = min(delta_e / 10.0, 1.0)
        
        # Texture: 0-50 scale (material-dependent, weighted combination)
        norm_texture = min(texture_delta / 50.0, 1.0)
        
        # Gloss: 0-50 LAB L* difference scale
        norm_gloss = min(gloss_delta / 50.0, 1.0)
        
        # Perceptos Index uses adaptive weighting based on significance
        if delta_e > 6.0:  # Significant color difference
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
        
        # Enhanced human perception logic using Perceptos Index
        # High Perceptos Index (85+) = Excellent uniformity
        # Medium Perceptos Index (70-84) = Acceptable variation
        # Low Perceptos Index (<70) = Noticeable differences
        
        if avg_perceptos >= 85 and uniformity_percentage >= 80 and major_color_issues <= 1:
            assessment = "uniform"
        elif avg_perceptos >= 70 and uniformity_percentage >= 60:
            assessment = "acceptable"
        elif major_color_issues >= total_points * 0.4 or avg_color > 7.0 or avg_perceptos < 60:
            assessment = "non-uniform"
        else:
            assessment = "acceptable"
            
        return assessment
    
    def analyze_points(self, original_path, replacement_path, original_points, replacement_points, 
                      measurement_type='standard', measurement_distance='18 inches'):
        """Analyze corresponding points between two images"""
        try:
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
            
            logger.info(f"Analysis completed for {len(results)} point pairs using {measurement_type} at {measurement_distance}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing points: {str(e)}")
            raise
    
    def generate_pdf_report(self, session_id, results, reports_folder, 
                           measurement_type='standard', measurement_distance='18 inches'):
        """Generate PDF report using ReportLab"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optikos_report_{session_id}_{timestamp}.pdf"
            filepath = os.path.join(reports_folder, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title with gamified styling
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=20,
                textColor=colors.HexColor('#1E3A8A'),  # Deep blue
                alignment=1  # Center alignment
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=30,
                textColor=colors.HexColor('#059669'),  # Green accent
                alignment=1  # Center alignment
            )
            
            story.append(Paragraph("Optikos | Visual Uniformity Analysis Report", title_style))
            story.append(Paragraph('Professional Material Analysis', subtitle_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Measurement Information Section
            measurement_style = ParagraphStyle(
                'MeasurementInfo',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=15,
                textColor=colors.HexColor('#374151'),
                backColor=colors.HexColor('#F3F4F6'),
                borderPadding=10,
                borderWidth=1,
                borderColor=colors.HexColor('#D1D5DB')
            )
            
            measurement_info = f"""
            <b>Measurement Configuration:</b><br/>
            <b>Type:</b> {measurement_type.title()}<br/>
            <b>Distance:</b> {measurement_distance}<br/>
            <b>Analysis Date:</b> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
            """
            story.append(Paragraph(measurement_info, measurement_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Summary with enhanced styling
            summary_style = ParagraphStyle(
                'SummaryHeader',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.HexColor('#1F2937')
            )
            story.append(Paragraph("Analysis Summary", summary_style))
            
            total_points = len(results)
            uniform_points = sum(1 for r in results if r['is_uniform'])
            avg_perceptos = round(sum(r['perceptos_index'] for r in results) / total_points, 2)
            
            # Professional summary with clear color coding
            status_color = colors.HexColor('#10B981') if uniform_points == total_points else colors.HexColor('#F59E0B')
            overall_status = 'UNIFORM' if uniform_points == total_points else 'NON-UNIFORM'
            
            summary_text = f"""
            <b>Total Points Analyzed:</b> {total_points}<br/>
            <b>Uniform Points:</b> {uniform_points}<br/>
            <b>Non-Uniform Points:</b> {total_points - uniform_points}<br/>
            <b>Average Perceptos Index:</b> {avg_perceptos}/100<br/>
            <b>Overall Assessment:</b> <font color="{status_color.hexval()}">{overall_status}</font>
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Results with professional styling
            results_style = ParagraphStyle(
                'ResultsHeader',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.HexColor('#1F2937')
            )
            story.append(Paragraph("Detailed Point Analysis", results_style))
            
            # Calculate human perception assessment
            assessment = self.calculate_human_perception_assessment(results)
            
            # Professional assessment display
            if assessment == "uniform":
                overall_status = 'UNIFORM'
                achievement_color = colors.HexColor('#059669')
            elif assessment == "acceptable":
                overall_status = 'ACCEPTABLE'
                achievement_color = colors.HexColor('#0EA5E9')
            else:
                overall_status = 'NON-UNIFORM'
                achievement_color = colors.HexColor('#EF4444')
            
            # Professional summary with human perception logic
            summary_text = f"""
            <b>Total Points Analyzed:</b> {total_points}<br/>
            <b>Uniform Points:</b> {uniform_points}<br/>
            <b>Non-Uniform Points:</b> {total_points - uniform_points}<br/>
            <b>Average Perceptos Index:</b> {avg_perceptos}/100<br/>
            <b>Overall Assessment:</b> <font color="{achievement_color.hexval()}">{overall_status}</font>
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Results with improved table formatting
            results_style = ParagraphStyle(
                'ResultsHeader',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.HexColor('#1F2937')
            )
            story.append(Paragraph("Detailed Point Analysis", results_style))
            
            # Create enhanced table data with better column widths
            table_data = [
                ['Point #', 'Color', 'ΔE', 'Texture', 'Gloss', 'Perceptos Index', 'Status']
            ]
            
            for result in results:
                # Enhanced status with color coding
                if result['is_uniform']:
                    status = "Uniform"
                else:
                    status = "Non-Uniform"
                
                table_data.append([
                    str(result['point_number']),
                    result['color'].title(),
                    str(result['delta_e']),
                    str(result['texture_delta']),
                    str(result['gloss_delta']),
                    str(result['perceptos_index']),
                    status
                ])
            
            # Create enhanced table with optimized column widths
            table = Table(table_data, colWidths=[0.7*inch, 0.8*inch, 0.7*inch, 0.9*inch, 0.7*inch, 1.0*inch, 1.2*inch])
            table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F9FAFB'), colors.HexColor('#FFFFFF')]),
                
                # Borders and grid
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#D1D5DB')),
                
                # Padding
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.4*inch))
            
            # Enhanced Technical Notes with emojis and modern styling
            tech_style = ParagraphStyle(
                'TechHeader',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.HexColor('#1F2937')
            )
            story.append(Paragraph("Technical Notes & Methodology", tech_style))
            
            tech_notes = """
            <b>Delta E (ΔE):</b> Color difference using CIE2000 formula. Values > 5 indicate non-uniform color.<br/>
            <b>Texture Delta:</b> Combined LBP, HOG, and Gabor filter analysis. Values > 20 indicate texture variation.<br/>
            <b>Gloss Delta:</b> Brightness difference in LAB color space. Values > 15 indicate gloss variation.<br/>
            <b>Perceptos Index:</b> Weighted composite score (0-100) combining all factors for material uniformity assessment.<br/>
            <b>Measurement Standards:</b> Analysis performed according to professional material comparison protocols.
            """
            story.append(Paragraph(tech_notes, styles['Normal']))
            
            # Add footer with branding
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=0,
                textColor=colors.HexColor('#6B7280'),
                alignment=1  # Center alignment
            )
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Generated by Optikos Material Analysis Platform | Professional Material Uniformity Assessment", footer_style))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
