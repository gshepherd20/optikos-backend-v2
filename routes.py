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
        
        # Calculate human perception assessment
        assessment = analyzer.calculate_human_perception_assessment(results)
        
        # Add overall summary metrics
        total_points = len(results)
        uniform_points = sum(1 for r in results if r['is_uniform'])
        avg_perceptos = round(sum(r['perceptos_index'] for r in results) / total_points, 2) if results else 0
        
        logger.info(f"Analysis completed for session {session_id}")
        
        return jsonify({
            'results': results,
            'session_id': session_id,
            'measurement_type': measurement_type,
            'measurement_distance': measurement_distance,
            'summary': {
                'total_points': total_points,
                'uniform_points': uniform_points,
                'average_perceptos': avg_perceptos,
                'human_assessment': assessment
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing points: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        results = data.get('results', [])
        measurement_type = data.get('measurement_type', 'standard')
        measurement_distance = data.get('measurement_distance', '18 inches')
        
        if not session_id or not results:
            return jsonify({'error': 'Session ID and results are required'}), 400
        
        analyzer = MaterialAnalyzer()
        report_path = analyzer.generate_pdf_report(
            session_id, 
            results, 
            app.config['REPORTS_FOLDER'],
            measurement_type=measurement_type,
            measurement_distance=measurement_distance
        )
        
        logger.info(f"PDF report generated for session {session_id}")
        
        return jsonify({
            'report_url': url_for('download_report', filename=os.path.basename(report_path)),
            'message': 'Report generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500

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

@app.route('/mobile/analyze', methods=['POST'])
def mobile_analyze():
    """Mobile app analysis endpoint with real scientific calculations"""
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
            return jsonify({'error': 'Number of original and replacement points must match'}), 400
            
        if len(original_points) == 0:
            return jsonify({'error': 'At least one point pair is required'}), 400
        
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
        
        # Perform real scientific analysis
        analyzer = MaterialAnalyzer()
        results = analyzer.analyze_points(
            original_path, replacement_path, 
            original_points, replacement_points,
            measurement_type, measurement_distance
        )
        
        # Calculate summary statistics
        if results:
            avg_delta_e = sum(r['delta_e'] for r in results) / len(results)
            avg_texture = sum(r['texture_delta'] for r in results) / len(results)
            avg_gloss = sum(r['gloss_delta'] for r in results) / len(results)
            avg_perceptos = sum(r['perceptos_index'] for r in results) / len(results)
            
            # Determine overall assessment
            assessment = analyzer.calculate_human_perception_assessment(results)
            
            response_data = {
                'session_id': session_id,
                'analysis_results': {
                    'perceptosIndex': round(avg_perceptos, 1),
                    'colorDiff': round(avg_delta_e, 1),
                    'textureDiff': round(avg_texture, 2),
                    'glossDiff': round(avg_gloss, 2),
                    'status': assessment.capitalize(),
                    'numPoints': len(results)
                },
                'detailed_results': results,
                'measurement_config': {
                    'type': measurement_type,
                    'distance': measurement_distance
                }
            }
            
            logger.info(f"Mobile analysis completed for session {session_id} with {len(results)} points")
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Analysis failed to produce results'}), 500
            
    except Exception as e:
        logger.error(f"Error in mobile analysis: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

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
