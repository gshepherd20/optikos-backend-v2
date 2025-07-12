from app import app, initialize_app

# Initialize the app for WSGI
initialize_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
