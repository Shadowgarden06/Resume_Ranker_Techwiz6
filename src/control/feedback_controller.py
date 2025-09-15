# src/control/feedback_controller.py
from flask import Blueprint, request, jsonify, session
import os
import json
from datetime import datetime

feedback_bp = Blueprint('feedback', __name__, url_prefix='/feedback')

@feedback_bp.route('/submit', methods=['POST'])
def submit_feedback():
    """API to submit feedback and save to CV JSON"""
    try:
        print("📝 Received feedback request")
        
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})
        
        print(f" Request data: {data}")
        
        candidate_filename = data.get('candidate_filename')
        human_decision = data.get('human_decision')
        human_rating = data.get('human_rating')
        feedback_notes = data.get('feedback_notes', '')
        
        # Check required data
        if not candidate_filename:
            return jsonify({'success': False, 'error': 'Missing candidate_filename'})
        
        if not human_decision:
            return jsonify({'success': False, 'error': 'Missing human_decision'})
        
        # ✅ Use user-specific path instead of root uploads
        from services import get_user_uploads_dir
        user_dir = get_user_uploads_dir()
        cv_json_file = os.path.join(user_dir, f"{candidate_filename}.json")
        
        if not os.path.exists(cv_json_file):
            return jsonify({'success': False, 'error': f'CV JSON file not found: {cv_json_file}'})
        
        # Read current JSON file
        try:
            with open(cv_json_file, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading CV JSON: {str(e)}'})
        
        # Create feedback object
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'human_decision': human_decision,
            'human_rating': human_rating if human_rating else None,
            'feedback_notes': feedback_notes,
            'user_id': session.get('user_id', 'anonymous'),
            'session_id': session.get('session_id', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        }
        
        # Add feedback to CV data
        if 'feedback_history' not in cv_data:
            cv_data['feedback_history'] = []
        
        cv_data['feedback_history'].append(feedback_entry)
        
        # Update summary information
        cv_data['latest_feedback'] = {
            'decision': human_decision,
            'rating': human_rating if human_rating else None,
            'timestamp': datetime.now().isoformat(),
            'total_feedback': len(cv_data['feedback_history'])
        }
        
        # Save JSON file
        try:
            with open(cv_json_file, 'w', encoding='utf-8') as f:
                json.dump(cv_data, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Feedback saved to CV JSON: {candidate_filename}")
            
            return jsonify({
                'success': True, 
                'message': 'Feedback saved successfully!',
                'data': {
                    'candidate_filename': candidate_filename,
                    'human_decision': human_decision,
                    'timestamp': datetime.now().isoformat(),
                    'total_feedback': len(cv_data['feedback_history'])
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error writing CV JSON: {str(e)}'})
            
    except Exception as e:
        print(f"❌ Error in submit_feedback: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@feedback_bp.route('/get_cv_feedback/<candidate_filename>')
def get_cv_feedback(candidate_filename):
    """API to get feedback from CV JSON"""
    try:
        # ✅ Use user-specific path instead of root uploads
        from services import get_user_uploads_dir
        user_dir = get_user_uploads_dir()
        cv_json_file = os.path.join(user_dir, f"{candidate_filename}.json")
        
        if not os.path.exists(cv_json_file):
            return jsonify({'success': True, 'feedback_history': [], 'latest_feedback': None})
        
        with open(cv_json_file, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        
        feedback_history = cv_data.get('feedback_history', [])
        latest_feedback = cv_data.get('latest_feedback', None)
        
        return jsonify({
            'success': True,
            'feedback_history': feedback_history,
            'latest_feedback': latest_feedback
        })
        
    except Exception as e:
        print(f"❌ Error getting CV feedback: {e}")
        return jsonify({'success': False, 'error': str(e)})

@feedback_bp.route('/test')
def test_feedback():
    """Test endpoint to check feedback system"""
    try:
        return jsonify({
            'success': True,
            'message': 'Feedback system is working!',
            'uploads_dir': os.path.exists("uploads"),
            'sample_files': len([f for f in os.listdir("uploads") if f.endswith('.json')]) if os.path.exists("uploads") else 0
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

