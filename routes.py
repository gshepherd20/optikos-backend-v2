import os
import json
import uuid
import base64
from flask import render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from app import app
from analysis import MaterialAnalyzer
import logging

# Enable CORS for mobile app integration
CORS(app)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        if 'original_image' not in request.files or 'replacement_image' not in request.files:
            return jsonify({'error': 'Both original and replacement images are required'}), 400
        
        original_file = request.files['original_image']
        replacement_file = request.files['replacement_image']
        
        if original_file.filename == '' or replacement_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(original_file.filename) and allowed_file(replacement_file.filename)):
            return jsonify({'error': 'Invalid file type. Please use PNG, JPG, JPEG, GIF, or BMP files'}), 400
        
        # Generate unique session ID for this comparison
        session_id = str(uuid.uuid4())
        
        # Save uploaded files
        original_filename = secure_filename(f"{session_id}_original_{original_file.filename}")
        replacement_filename = secure_filename(f"{session_id}_replacement_{replacement_file.filename}")
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        replacement_path = os.path.join(app.config['UPLOAD_FOLDER'], replacement_filename)
        
        original_file.save(original_path)
        replacement_file.save(replacement_path)
        
        logger.info(f"Images uploaded successfully for session {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'original_image': original_filename,
            'replacement_image': replacement_filename,
            'message': 'Images uploaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        return jsonify({'error': 'Failed to upload images'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/analyze', methods=['POST'])
def analyze_points():
    try:
        data = request.get_json()
        
        session_id = data.get('session_id')
        original_points = data.get('original_points', [])
        replacement_points = data.get('replacement_points', [])
        measurement_type = data.get('measurement_type', 'standard')
        measurement_distance = data.get('measurement_distance', '18 inches')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if len(original_points) != len(replacement_points):
            return jsonify({'error': 'Number of points must match on both images'}), 400
        
        if len(original_points) == 0:
            return jsonify({'error': 'At least one point pair is required'}), 400
        
        # Find the uploaded images
        original_filename = None
        replacement_filename = None
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(f"{session_id}_original_"):
                original_filename = filename
            elif filename.startswith(f"{session_id}_replacement_"):
                replacement_filename = filename
        
        if not original_filename or not replacement_filename:
            return jsonify({'error': 'Images not found for this session'}), 404
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        replacement_path = os.path.join(app.config['UPLOAD_FOLDER'], replacement_filename)
        
        # Perform analysis
        analyzer = MaterialAnalyzer()
        results = analyzer.analyze_points(
            original_path, 
            replacement_path, 
            original_points, 
            replacement_points,
            measurement_type=measurement_type,
            measurement_distance=measurement_distance
        )
        
        # Calculate photo alignment analysis
        original_img = analyzer.load_and_resize_image(original_path)
        replacement_img = analyzer.load_and_resize_image(replacement_path)
        alignment_analysis = analyzer.calculate_photo_alignment_score(
            original_img, replacement_img, original_points, replacement_points
        )
        
        # Calculate human perception assessment
        assessment = analyzer.calculate_human_perception_assessment(results)
        
        # Add overall summary metrics
        total_points = len(results)
        uniform_points = sum(1 for r in results if r['is_uniform'])
        avg_perceptos = round(sum(r['perceptos_index'] for r in results) / total_points, 2) if results else 0
        
        logger.info(f"Analysis completed for session {session_id}")
        logger.info(f"Photo Alignment Score: {alignment_analysis['score']:.1f} ({alignment_analysis['assessment']})")
        
        return jsonify({
            'results': results,
            'session_id': session_id,
            'measurement_type': measurement_type,
            'measurement_distance': measurement_distance,
            'alignment_analysis': alignment_analysis,
            'summary': {
                'total_points': total_points,
                'uniform_points': uniform_points,
                'average_perceptos': avg_perceptos,
                'human_assessment': assessment,
                'photo_alignment_score': alignment_analysis['score'],
                'photo_alignment_assessment': alignment_analysis['assessment']
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing points: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        logger.info(f"=== REPORT GENERATION DEBUG ===")
        logger.info(f"Raw request data: {json.dumps(data, indent=2)}")
        
        session_id = data.get('session_id')
        results = data.get('results', {})
        measurement_type = data.get('measurement_type', 'standard')
        measurement_distance = data.get('measurement_distance', '18 inches')
        
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Results type: {type(results)}")
        logger.info(f"Results content: {results}")
        logger.info(f"Results is list: {isinstance(results, list)}")
        logger.info(f"Results is dict: {isinstance(results, dict)}")
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Handle mobile app format - results should be a list of point data
        if isinstance(results, list):
            # Mobile app sends results as array directly
            report_data = results
            logger.info(f"Using results as direct array, length: {len(results)}")
        elif isinstance(results, dict) and 'point_data' in results:
            # Legacy format with point_data wrapper
            report_data = results['point_data']
            logger.info(f"Extracted point_data array, length: {len(report_data)}")
        else:
            # Fallback
            report_data = []
            logger.warning(f"No valid point data found, using empty array")
        
        logger.info(f"Final report_data: {report_data}")
        
        if len(report_data) == 0:
            return jsonify({'error': 'No analysis points provided for report generation'}), 400
        
        analyzer = MaterialAnalyzer()
        report_path = analyzer.generate_pdf_report(
            session_id, 
            report_data, 
            app.config['REPORTS_FOLDER'],
            measurement_type=measurement_type,
            measurement_distance=measurement_distance
        )
        
        logger.info(f"PDF report generated for session {session_id}")
        
        return jsonify({
            'success': True,
            'report_filename': os.path.basename(report_path),
            'report_url': url_for('download_report', filename=os.path.basename(report_path)),
            'message': 'Report generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

@app.route('/reports/<filename>')
def download_report(filename):
    return send_file(os.path.join(app.config['REPORTS_FOLDER'], filename), as_attachment=True)

@app.route('/mobile/upload', methods=['POST'])
def mobile_upload():
    """Mobile app image upload endpoint with base64 support"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        original_b64 = data.get('original_image')
        replacement_b64 = data.get('replacement_image')
        
        if not original_b64 or not replacement_b64:
            return jsonify({'error': 'Both original and replacement images are required'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Decode base64 images and save
        def save_b64_image(b64_data, prefix):
            # Remove data URL prefix if present
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            
            image_data = base64.b64decode(b64_data)
            filename = f"{session_id}_{prefix}.jpg"
            filepath = os.path.join('uploads', filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            return filename
        
        original_filename = save_b64_image(original_b64, 'original')
        replacement_filename = save_b64_image(replacement_b64, 'replacement')
        
        logger.info(f"Mobile images uploaded successfully for session {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'original_image': original_filename,
            'replacement_image': replacement_filename,
            'message': 'Images uploaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error uploading mobile images: {str(e)}")
        return jsonify({'error': 'Failed to upload images'}), 500

@app.route('/find_corresponding_point', methods=['POST'])
def find_corresponding_point():
    """Find corresponding point in second image based on click in first image"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        click_x = data.get('x')
        click_y = data.get('y')
        image_type = data.get('image_type', 'original')  # which image was clicked
        
        if not session_id or click_x is None or click_y is None:
            return jsonify({'error': 'Session ID and coordinates are required'}), 400
        
        # Find the uploaded images
        original_filename = None
        replacement_filename = None
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(f"{session_id}_original_"):
                original_filename = filename
            elif filename.startswith(f"{session_id}_replacement_"):
                replacement_filename = filename
        
        if not original_filename or not replacement_filename:
            return jsonify({'error': 'Images not found for this session'}), 404
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        replacement_path = os.path.join(app.config['UPLOAD_FOLDER'], replacement_filename)
        
        # Load images
        analyzer = MaterialAnalyzer()
        original_img = analyzer.load_and_resize_image(original_path)
        replacement_img = analyzer.load_and_resize_image(replacement_path)
        
        # Find corresponding point
        if image_type == 'original':
            # Click was on original image, find corresponding point in replacement
            corresponding = analyzer.find_corresponding_point(
                original_img, replacement_img, click_x, click_y
            )
        else:
            # Click was on replacement image, find corresponding point in original
            corresponding = analyzer.find_corresponding_point(
                replacement_img, original_img, click_x, click_y
            )
        
        if corresponding:
            logger.info(f"Found corresponding point: ({corresponding['x']}, {corresponding['y']}) with confidence {corresponding['confidence']:.2f}")
            return jsonify({
                'success': True,
                'corresponding_point': corresponding,
                'clicked_image': image_type,
                'target_image': 'replacement' if image_type == 'original' else 'original'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not find corresponding point',
                'clicked_image': image_type
            })
            
    except Exception as e:
        logger.error(f"Error finding corresponding point: {str(e)}")
        return jsonify({'error': 'Failed to find corresponding point'}), 500

@app.route('/mobile/analyze', methods=['POST'])
def mobile_analyze():
    """Mobile app analysis endpoint with toggle for manual points or auto clustering"""
    try:
        data = request.get_json()
        
        session_id = data.get('session_id')
        analysis_mode = data.get('analysis_mode', 'manual')  # 'manual' or 'cluster'
        measurement_type = data.get('measurement_type', 'standard')
        measurement_distance = data.get('measurement_distance', '18 inches')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
            
        # Validate based on analysis mode
        if analysis_mode == 'manual':
            original_points = data.get('original_points', [])
            replacement_points = data.get('replacement_points', [])
            
            if len(original_points) != len(replacement_points):
                return jsonify({'error': 'Number of original and replacement points must match'}), 400
                
            if len(original_points) == 0:
                return jsonify({'error': 'At least one point pair is required'}), 400
        elif analysis_mode == 'cluster':
            n_clusters = data.get('n_clusters', 4)
            if n_clusters < 2 or n_clusters > 8:
                return jsonify({'error': 'Number of clusters must be between 2 and 8'}), 400
        
        # Find uploaded images
        original_filename = None
        replacement_filename = None
        
        for filename in os.listdir('uploads'):
            if filename.startswith(session_id):
                if 'original' in filename:
                    original_filename = filename
                elif 'replacement' in filename:
                    replacement_filename = filename
        
        if not original_filename or not replacement_filename:
            return jsonify({'error': 'Images not found for session'}), 404
            
        original_path = os.path.join('uploads', original_filename)
        replacement_path = os.path.join('uploads', replacement_filename)
        
        # Perform analysis based on selected mode
        analyzer = MaterialAnalyzer()
        
        if analysis_mode == 'cluster':
            # Auto cluster analysis
            results = analyzer.analyze_clusters(
                original_path, replacement_path,
                n_clusters=n_clusters,
                measurement_type=measurement_type,
                measurement_distance=measurement_distance
            )
        else:
            # Manual point analysis
            results = analyzer.analyze_points(
                original_path, replacement_path, 
                original_points, replacement_points,
                measurement_type, measurement_distance
            )
        
        # Extract results based on analysis mode
        if analysis_mode == 'cluster':
            # For cluster analysis, use the structured results
            point_results = results.get('detailed_results', [])
            lighting_validation = results.get('lighting_validation', {})
            photo_alignment = {'score': 100}  # Clusters don't need alignment scoring
            analysis_results = results.get('analysis_results', {})
        else:
            # For manual point analysis, calculate photo alignment
            original_img = analyzer.load_and_resize_image(original_path)
            replacement_img = analyzer.load_and_resize_image(replacement_path)
            alignment_analysis = analyzer.calculate_photo_alignment_score(
                original_img, replacement_img, original_points, replacement_points
            )
            
            point_results = results.get('point_analysis', [])
            lighting_validation = results.get('lighting_validation', {})
            photo_alignment = results.get('photo_alignment', {})
        
        # Calculate summary statistics based on analysis mode
        if point_results:
            if analysis_mode == 'cluster':
                # Use pre-calculated cluster results
                avg_delta_e = analysis_results.get('average_delta_e', 0)
                avg_texture = 0  # Texture delta not used in cluster mode
                avg_gloss = 0   # Gloss delta not used in cluster mode
                avg_perceptos = analysis_results.get('perceptos_index', 0)
                assessment = analysis_results.get('uniformity_assessment', 'uniform')
                
                response_data = {
                    'session_id': session_id,
                    'analysis_mode': analysis_mode,
                    'analysis_results': {
                        'perceptosIndex': round(avg_perceptos, 1),
                        'colorDiff': round(avg_delta_e, 1),
                        'photoAlignment': 100,  # Clusters don't need alignment
                        'textureDiff': 0,
                        'glossDiff': 0,
                        'status': assessment.capitalize(),
                        'numClusters': analysis_results.get('points_analyzed', 4),
                        'confidencePercentage': analysis_results.get('confidence_percentage', 80)
                    },
                    'detailed_results': point_results,
                    'cluster_data': results.get('cluster_data', {}),
                    'lighting_validation': lighting_validation,
                    'verified_report_eligible': lighting_validation.get('verified_report_eligible', False),
                    'measurement_config': {
                        'type': measurement_type,
                        'distance': measurement_distance
                    }
                }
            else:
                # Manual point analysis calculations
                avg_delta_e = sum(r['delta_e'] for r in point_results) / len(point_results)
                avg_texture = sum(r['texture_delta'] for r in point_results) / len(point_results)
                avg_gloss = sum(r['gloss_delta'] for r in point_results) / len(point_results)
                avg_perceptos = sum(r['perceptos_index'] for r in point_results) / len(point_results)
                
                # Determine overall assessment
                assessment = analyzer.calculate_human_perception_assessment(point_results)
                
                response_data = {
                    'session_id': session_id,
                    'analysis_mode': analysis_mode,
                    'analysis_results': {
                        'perceptosIndex': round(avg_perceptos, 1),
                        'colorDiff': round(avg_delta_e, 1),
                        'photoAlignment': photo_alignment.get('score', 0),
                        'textureDiff': round(avg_texture, 2),
                        'glossDiff': round(avg_gloss, 2),
                        'status': assessment.capitalize(),
                        'numPoints': len(point_results)
                    },
                    'detailed_results': point_results,
                    'lighting_validation': lighting_validation,
                    'verified_report_eligible': lighting_validation.get('verified_report_eligible', False),
                    'measurement_config': {
                        'type': measurement_type,
                        'distance': measurement_distance
                    }
                }
            
            logger.info(f"Mobile analysis completed for session {session_id} with {len(point_results)} points")
            logger.info(f"Lighting validation: {lighting_validation.get('overall_lighting_quality', 'unknown')}, Verified eligible: {lighting_validation.get('verified_report_eligible', False)}")
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Analysis failed to produce results'}), 500
            
    except Exception as e:
        logger.error(f"Error in mobile analysis: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/mobile/generate_report', methods=['POST'])
def mobile_generate_report():
    """Generate PDF report for mobile app"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        measurement_type = data.get('measurement_type', 'standard')
        measurement_distance = data.get('measurement_distance', '18 inches')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Get the actual analysis data passed from mobile app
        analysis_data = data.get('analysis_data')
        
        if not analysis_data:
            return jsonify({'error': 'Analysis data is required for report generation'}), 400
        
        try:
            # Use the real analysis data from mobile app
            # The data comes from the actual MaterialAnalyzer.analyze_points() call
            report_data = analysis_data
            
            # Debug logging for data structure verification
            logger.info(f"Report data type: {type(report_data)}")
            logger.info(f"Report data keys: {list(report_data.keys()) if isinstance(report_data, dict) else 'Not a dict'}")
            
            if 'point_analysis' in report_data:
                logger.info(f"Point analysis type: {type(report_data['point_analysis'])}")
                logger.info(f"Point analysis length: {len(report_data['point_analysis']) if isinstance(report_data['point_analysis'], list) else 'Not a list'}")
                if isinstance(report_data['point_analysis'], list) and len(report_data['point_analysis']) > 0:
                    logger.info(f"First point keys: {list(report_data['point_analysis'][0].keys()) if isinstance(report_data['point_analysis'][0], dict) else 'First point not a dict'}")
            
            analyzer = MaterialAnalyzer()
            report_path = analyzer.generate_pdf_report(
                session_id, 
                report_data, 
                'reports',
                measurement_type=measurement_type,
                measurement_distance=measurement_distance
            )
            
            logger.info(f"Mobile PDF report generated for session {session_id}")
            
            return jsonify({
                'success': True,
                'report_filename': os.path.basename(report_path),
                'download_url': f"https://web-production-67f3.up.railway.app/reports/{os.path.basename(report_path)}",
                'message': 'Report generated successfully'
            })
            
        except Exception as e:
            logger.error(f"Error generating mobile report: {str(e)}")
            return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Error in mobile report generation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500
