import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def health_check():
    return jsonify({"status": "ok", "message": "Optikos Backend API - Minimal Version"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "minimal"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
