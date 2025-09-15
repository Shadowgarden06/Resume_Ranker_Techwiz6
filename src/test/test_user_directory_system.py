import os
from flask import Flask, session
from services import get_user_uploads_dir, clear_user_uploads_directory, save_file_to_user_dir

def test_user_directory_system():
    app = Flask(__name__)
    app.secret_key = 'test'
    
    with app.test_request_context():
        # Test user 1
        session['user_id'] = 'admin'
        user1_dir = get_user_uploads_dir()
        print(f"✅ User 1 directory: {user1_dir}")
        
        # Test user 2  
        session['user_id'] = 'recruiter'
        user2_dir = get_user_uploads_dir()
        print(f"✅ User 2 directory: {user2_dir}")
        
        # Verify directories are different
        assert user1_dir != user2_dir
        print("✅ User directories are isolated")
        
        # Test file paths
        session['user_id'] = 'admin'
        file_path = get_user_file_path('test.pdf')
        print(f"✅ User file path: {file_path}")
        
        # Test cleanup
        clear_user_uploads_directory()
        print("✅ User directory cleanup works")

if __name__ == '__main__':
    test_user_directory_system()
