from functools import wraps
from flask import session, redirect, url_for


USERS = {
    'admin': 'admin123',
    'recruiter': 'recruiter123',
    'hr': 'hr123'
}


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login_bp.login'))
        return f(*args, **kwargs)
    return decorated_function


