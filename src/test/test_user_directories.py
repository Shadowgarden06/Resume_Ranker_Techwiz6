import os
import tempfile
from flask import Flask, session
from services import get_user_uploads_dir, clear_user_uploads_directory

def test_user_directories():
    app = Flask(__name__)
    app.secret_key = 'test'
    
    with app.test_request_context():
        # Test user 1
        session['user_id'] = 'user1'
        user1_dir = get_user_uploads_dir()
        print(f"User 1 directory: {user1_dir}")
        
        # Test user 2  
        session['user_id'] = 'user2'
        user2_dir = get_user_uploads_dir()
        print(f"User 2 directory: {user2_dir}")
        
        # Verify directories are different
        assert user1_dir != user2_dir
        print("✅ User directories are isolated")

if __name__ == '__main__':
    test_user_directories()
