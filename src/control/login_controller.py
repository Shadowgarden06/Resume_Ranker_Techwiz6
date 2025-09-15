# moved from src/login_controller.py
from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from auth import USERS
from services import clear_user_uploads_directory, clear_session_data


bp = Blueprint('login_bp', __name__)


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            # Clear old data for this user (does not affect other users)
            clear_user_uploads_directory()
            clear_session_data()
            
            # Set user_id AFTER clearing session
            session['user_id'] = username
            session['logged_in'] = True
            
            flash('Login successful!', 'success')
            return redirect(url_for('index_bp.index'))
        flash('Invalid username or password!', 'error')
    return render_template('login.html')


@bp.route('/logout')
def logout():
    """Logout and only delete current user's data"""
    try:
        # Delete current user's files (does not affect other users)
        clear_user_uploads_directory()
        
        # Delete current user's session data
        clear_session_data()
        
        flash('You have been logged out. Your data has been cleared.', 'info')
        return redirect(url_for('login_bp.login'))
        
    except Exception as e:
        print(f"Error during logout: {e}")
        # Still logout even if there's an error
        session.clear()
        flash('You have been logged out. Some data may not have been cleared completely.', 'warning')
        return redirect(url_for('login_bp.login'))


