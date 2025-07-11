from app import app

# Everything else handled in app.py to prevent double registration

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
