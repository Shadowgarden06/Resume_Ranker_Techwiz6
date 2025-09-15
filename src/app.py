from flask import Flask
from flask_session import Session
import os
from dotenv import load_dotenv  # ✅ Add this import

# ✅ Load environment variables from .env file
load_dotenv()

from document_qa import initialize_qa_system
from services import load_skills
from control.index_controller import bp as index_bp
from control.login_controller import bp as login_bp
from control.search_controller import bp as search_bp
from control.uploads_controller import bp as uploads_bp
from control.qa_home_controller import bp as qa_bp

# Import feedback controller
try:
    from control.feedback_controller import feedback_bp
    FEEDBACK_AVAILABLE = True
    print("✅ Feedback controller imported successfully")
except Exception as e:
    print(f"⚠️ Warning: Feedback controller not available: {e}")
    FEEDBACK_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

# Register blueprints (controllers)
app.register_blueprint(index_bp)
app.register_blueprint(login_bp)
app.register_blueprint(search_bp)   
app.register_blueprint(uploads_bp)
app.register_blueprint(qa_bp)

# Register feedback blueprint if available
if FEEDBACK_AVAILABLE:
    app.register_blueprint(feedback_bp)
    print("✅ Feedback blueprint registered")

from flask import send_from_directory

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js')

if __name__ == '__main__':
    # Debug: Check if API key is loaded
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ GOOGLE_API_KEY loaded: {api_key[:10]}...")
    else:
        print("❌ GOOGLE_API_KEY not found in environment")
    
    # Initialize shared services
    load_skills()
    try:
        initialize_qa_system()
        print("✅ Document QA system initialized successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Document QA system not initialized: {e}")
        print("💡 Document QA is optional. You can set GOOGLE_API_KEY environment variable to enable it.")
    app.run(host="0.0.0.0", port=8080, debug=True)
