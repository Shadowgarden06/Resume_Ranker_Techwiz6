# moved from src/qa_home_controller.py
import os
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from langdetect import detect
from deep_translator import GoogleTranslator
from auth import login_required
from document_qa import initialize_qa_system, get_qa_system


bp = Blueprint('qa_bp', __name__)


def ensure_qa_system():
    """Ensure QA system is available, return None if not"""
    try:
        return get_qa_system()
    except Exception:
        try:
            initialize_qa_system()
            return get_qa_system()
        except Exception as e:
            print(f"⚠️ QA system initialization failed: {e}")
            return None

@bp.route('/qa_home')
@login_required
def qa_home():
    """Document QA home page"""
    qa_error = None
    qa = ensure_qa_system()
    if qa is None:
        qa_error = 'Document QA system is not available. This feature requires GOOGLE_API_KEY environment variable.'
    documents = qa.get_document_list() if qa else []
    return render_template('qa_home.html', documents=documents, qa_error=qa_error)


@bp.route('/qa/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        qa = ensure_qa_system()
        if qa is None:
            return jsonify({'success': False, 'message': 'QA system not initialized. Please set GOOGLE_API_KEY.'})
        question = request.form.get('question', '').strip()
        document_id = request.form.get('document_id', '').strip()
        if not question:
            return jsonify({'success': False, 'message': 'Please enter a question.'})
        result = qa.ask_question(question, document_id if document_id and document_id != 'all' else None)

        # If question is in Vietnamese, translate the answer to Vietnamese
        try:
            lang = detect(question)
            if lang.startswith('vi') and result.get('answer'):
                translator = GoogleTranslator(source='en', target='vi')
                translated = translator.translate(result['answer'])
                result['answer'] = translated
        except Exception:
            pass
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'document_id': result.get('document_id', 'multiple')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@bp.route('/qa/add_document', methods=['POST'])
@login_required
def add_document_to_qa():
    try:
        print(f" Adding document to QA...")
        
        qa = ensure_qa_system()
        if qa is None:
            print("❌ QA system not initialized")
            return jsonify({'success': False, 'message': 'QA system not initialized. Please set GOOGLE_API_KEY.'})
        
        filename = request.form.get('filename', '').strip()
        print(f"📁 Filename: {filename}")
        
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided.'})
        
        # Use user directory system
        from services import get_user_file_path
        file_path = get_user_file_path(filename)
        print(f"📂 File path: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return jsonify({'success': False, 'message': f'File not found: {file_path}'})
        
        print(f"✅ File found, adding to QA...")
        document_id = qa.add_document(file_path, filename)
        print(f"✅ Document added with ID: {document_id}")
        
        return jsonify({'success': True, 'message': f'Document {filename} added successfully.', 'document_id': document_id})
        
    except Exception as e:
        print(f"❌ Error adding document to QA: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@bp.route('/qa/remove_document', methods=['POST'])
@login_required
def remove_document_from_qa():
    try:
        qa = ensure_qa_system()
        if qa is None:
            return jsonify({'success': False, 'message': 'QA system not initialized. Please set GOOGLE_API_KEY.'})
        document_id = request.form.get('document_id', '').strip()
        if not document_id:
            return jsonify({'success': False, 'message': 'No document ID provided.'})
        success = qa.remove_document(document_id)
        if success:
            return jsonify({'success': True, 'message': f'Document {document_id} removed successfully.'})
        return jsonify({'success': False, 'message': 'Document not found.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@bp.route('/qa/documents')
@login_required
def qa_documents():
    try:
        qa = ensure_qa_system()
        if qa is None:
            return jsonify({'success': False, 'message': 'QA system not initialized. Please set GOOGLE_API_KEY.'})
        documents = qa.get_document_list()
        return jsonify({'success': True, 'documents': documents})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@bp.route('/get_available_files')
@login_required
def get_available_files():
    """Get list of available files for QA system"""
    try:
        from services import list_user_files
        
        files = list_user_files()
        file_list = []
        
        for file_info in files:
            filename = file_info['filename']
            # Only include PDF and DOCX files
            if filename.lower().endswith(('.pdf', '.docx')):
                file_list.append({
                    'filename': filename,
                    'size': file_info['size'],
                    'created': file_info['created'].strftime('%Y-%m-%d %H:%M')
                })
        
        return jsonify({
            'success': True,
            'files': file_list
        })
        
    except Exception as e:
        print(f"❌ Error getting available files: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


