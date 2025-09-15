# moved from src/uploads_controller.py
import os
import json
import re
from datetime import datetime
from flask import Blueprint, render_template, jsonify, send_from_directory, abort
from auth import login_required
from services import (
    extract_text, extract_entities_ner, extract_basic_entities,
    extract_skills_spacy, save_cv_to_session, load_uploaded_cvs_from_session,
    dedupe_cvs, clear_uploads_directory, clear_session_data,
    get_user_uploads_dir, save_file_to_user_dir, get_user_file_path
)


bp = Blueprint('uploads_bp', __name__)


@bp.route('/upload_cv', methods=['POST'])
@login_required
def upload_cv():
    from flask import request
    if 'resume_files' not in request.files:
        return jsonify({'success': False, 'message': 'No files selected!'}), 400

    resume_files = request.files.getlist('resume_files')
    uploaded_count = 0

    # Create uploads directory for user
    user_dir = get_user_uploads_dir()

    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    for resume_file in resume_files:
        if resume_file.filename.lower().endswith(('.pdf', '.docx')):
            try:
                # Save to user directory
                resume_path = save_file_to_user_dir(resume_file, resume_file.filename)
                if resume_path:
                    # Extract information from CV
                    cv_data = extract_cv_data(resume_file.filename)
                    if cv_data:
                        # Save to session
                        save_cv_to_session(cv_data)
                        uploaded_count += 1
                        print(f"✅ Processed CV: {resume_file.filename}")
                    else:
                        print(f"❌ Failed to extract CV data: {resume_file.filename}")
            except Exception as e:
                print(f"Error processing file {resume_file.filename}: {e}")
                continue

    return jsonify({'success': True, 'message': f'Successfully uploaded {uploaded_count} CV(s)!'})


@bp.route('/upload_list')
@login_required
def upload_list():
    # ✅ Ensure session is refreshed from user directory
    from services import auto_reload_cvs_from_user_directory
    auto_reload_cvs_from_user_directory()
    
    cvs = load_uploaded_cvs_from_session()
    unique_cvs = dedupe_cvs()  # No parameter passed
    try:
        unique_cvs.sort(key=lambda x: x.get('upload_time',''), reverse=True)
    except Exception:
        pass
    
    return render_template('upload_list.html', cvs=unique_cvs)


@bp.route('/cv/<path:filename>')
@login_required
def cv_file(filename):
    # Use user-specific path instead of root uploads
    user_dir = get_user_uploads_dir()
    fpath = os.path.join(user_dir, filename)
    if os.path.isfile(fpath):
        return send_from_directory(user_dir, filename, as_attachment=True)
    abort(404)


@bp.route('/cv_text/<path:filename>')
@login_required
def cv_text(filename):
    # Use user-specific path instead of root uploads
    user_dir = get_user_uploads_dir()
    fpath = os.path.join(user_dir, filename)
    if not os.path.isfile(fpath):
        abort(404)
    text = extract_text(fpath) or ''
    meta = {'filename': filename, 'name': '', 'email': '', 'phone': '', 'upload_time': ''}
    from flask import session
    for rec in session.get('uploaded_cvs', []):
        jpath = rec.get('json_file_path')
        if jpath and os.path.exists(jpath):
            try:
                with open(jpath, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                if data.get('filename') == filename:
                    meta.update({
                        'name': data.get('name',''),
                        'email': data.get('email',''),
                        'phone': data.get('phone',''),
                        'upload_time': data.get('upload_time','')
                    })
                    break
            except Exception:
                pass
    return render_template('cv_view.html', text=text, meta=meta)


@bp.route('/cv_text_raw/<path:filename>')
@login_required
def cv_text_raw(filename):
    # Use user-specific path instead of root uploads
    user_dir = get_user_uploads_dir()
    fpath = os.path.join(user_dir, filename)
    if not os.path.isfile(fpath):
        return jsonify({'error': 'File not found'}), 404
    text = extract_text(fpath) or ''
    meta = {'filename': filename, 'name': '', 'email': '', 'phone': '', 'upload_time': ''}
    from flask import session
    for rec in session.get('uploaded_cvs', []):
        jpath = rec.get('json_file_path')
        if jpath and os.path.exists(jpath):
            try:
                with open(jpath, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                if data.get('filename') == filename:
                    meta.update({
                        'name': data.get('name',''),
                        'email': data.get('email',''),
                        'phone': data.get('phone',''),
                        'upload_time': data.get('upload_time','')
                    })
                    break
            except Exception:
                pass
    return jsonify({'meta': meta, 'text': text})


@bp.route('/delete_cv/<path:filename>', methods=['POST'])
@login_required
def delete_cv(filename):
    """Delete CV from system"""
    from services import delete_cv_from_session
    
    try:
        success = delete_cv_from_session(filename)
        if success:
            return jsonify({'success': True, 'message': f'CV {filename} deleted successfully!'})
        else:
            return jsonify({'success': False, 'message': f'Cannot delete CV {filename}'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting CV: {str(e)}'}), 500


@bp.route('/clear_all_uploads', methods=['POST'])
@login_required
def clear_all_uploads():
    """Delete all upload files (admin function)"""
    from services import clear_uploads_directory
    
    try:
        success = clear_uploads_directory()
        if success:
            return jsonify({'success': True, 'message': 'All upload files deleted successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Error occurred while deleting upload files'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@bp.route('/test_download')
def test_download():
    """Test route to check blueprint functionality"""
    return jsonify({'message': 'Uploads controller is working!'})

@bp.route('/download_upload_list_csv')
@login_required
def download_upload_list_csv():
    """Download CV uploads list as CSV with feedback information"""
    try:
        from services import load_uploaded_cvs_from_session
        
        # Get CV list from session
        cvs = load_uploaded_cvs_from_session()
        
        if not cvs:
            return jsonify({'success': False, 'message': 'No CVs available for download'}), 404
        
        # Create simple CSV
        import csv
        from io import StringIO
        from flask import Response
        from datetime import datetime
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = ['CV Name', 'Full Name', 'Email', 'Experience (years)']
        writer.writerow(headers)
        
        # Data
        for cv in cvs:
            row = [
                cv.get('filename', 'N/A'),
                cv.get('name', 'N/A'),
                cv.get('email', 'N/A'),
                cv.get('years_exp', 0)
            ]
            writer.writerow(row)
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_list_{timestamp}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error in download_upload_list_csv: {e}")
        return jsonify({'success': False, 'message': f'Error creating CSV: {str(e)}'}), 500

@bp.route('/download_upload_list_excel')
@login_required  
def download_upload_list_excel():
    """Download CV uploads list as Excel format"""
    try:
        from services import load_uploaded_cvs_from_session
        from datetime import datetime
        
        # Get CV list from session
        cvs = load_uploaded_cvs_from_session()
        
        if not cvs:
            return jsonify({'success': False, 'message': 'No CVs available for download'}), 404
        
        # Create Excel with full information (same as PDF)
        import pandas as pd
        from io import BytesIO
        from flask import Response
        
        data = []
        for cv in cvs:
            # Get feedback information
            from control.feedback_controller import get_cv_feedback_info
            feedback_info = get_cv_feedback_info(cv.get('filename', ''))
            
            # Format decision breakdown as string
            decision_breakdown = feedback_info.get('decision_breakdown', {})
            breakdown_text = ', '.join([f"{k}:{v}" for k, v in decision_breakdown.items()])
            
            data.append({
                'CV Name': cv.get('filename', 'N/A'),
                'Full Name': cv.get('name', 'N/A'),
                'Email': cv.get('email', 'N/A'),
                'Phone': cv.get('phone', 'N/A'),
                'Experience (years)': cv.get('years_exp', 0),
                'Skills': cv.get('skills_text', 'N/A'),
                'Upload Time': cv.get('upload_time', '')[:19] if cv.get('upload_time') else datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Total Feedback': feedback_info.get('total_feedback', 0),
                'Latest Decision': feedback_info.get('latest_decision', '') or '',
                'Average Rating': feedback_info.get('average_rating', '') or '',
                'Latest Feedback Time': feedback_info.get('latest_feedback_time', '') or '',
                'Decision Breakdown': breakdown_text
            })
        
        df = pd.DataFrame(data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Upload List', index=False)
            
            # Get worksheet for formatting
            worksheet = writer.sheets['Upload List']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Max width 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_list_{timestamp}.xlsx"
        
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error in download_upload_list_excel: {e}")
        return jsonify({'success': False, 'message': f'Error creating Excel: {str(e)}'}), 500

@bp.route('/download_upload_list_pdf')
@login_required
def download_upload_list_pdf():
    """Download CV uploads list as PDF format (Landscape)"""
    try:
        from services import load_uploaded_cvs_from_session
        
        # Get CV list from session
        cvs = load_uploaded_cvs_from_session()
        
        if not cvs:
            return jsonify({'success': False, 'message': 'No CVs available for download'}), 404
        
        # Create PDF with landscape mode
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import inch  # ✅ Add this import
        from io import BytesIO
        from flask import Response
        
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=landscape(A4),
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.3*inch, rightMargin=0.3*inch)
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("CV Uploads List", styles['Heading1'])
        
        # Table data (optimized for landscape)
        table_data = [['CV Name', 'Full Name', 'Email', 'Phone', 'Experience', 'Skills', 'Upload Time']]
        
        for cv in cvs:
            # Truncate long text
            cv_name = cv.get('filename', 'Unknown')[:25] + '...' if len(cv.get('filename', '')) > 25 else cv.get('filename', 'Unknown')
            full_name = cv.get('name', 'N/A')[:30] + '...' if len(cv.get('name', '')) > 30 else cv.get('name', 'N/A')
            email = cv.get('email', 'N/A')[:30] + '...' if len(cv.get('email', '')) > 30 else cv.get('email', 'N/A')
            phone = cv.get('phone', 'N/A')[:15] + '...' if len(cv.get('phone', '')) > 15 else cv.get('phone', 'N/A')
            skills = cv.get('skills_text', 'N/A')[:40] + '...' if len(cv.get('skills_text', '')) > 40 else cv.get('skills_text', 'N/A')
            
            table_data.append([
                cv_name,
                full_name,
                email,
                phone,
                f"{cv.get('years_exp', 0)}y",
                skills,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ])
        
        # Table with column widths optimized for landscape
        col_widths = [80, 100, 100, 60, 40, 120, 80]
        table = Table(table_data, colWidths=col_widths)
        
        # Table style
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), '#F2F2F2'),
            ('GRID', (0, 0), (-1, -1), 0.5, '#CCCCCC'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        # Build PDF
        doc.build([title, table])
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_list_{timestamp}.pdf"
        
        return Response(
            output.getvalue(),
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error in download_upload_list_pdf: {e}")
        return jsonify({'success': False, 'message': f'Error creating PDF: {str(e)}'}), 500


@bp.route('/debug_cvs')
@login_required
def debug_cvs():
    """Debug endpoint to check CV data"""
    from services import debug_user_directory, load_uploaded_cvs_from_session
    
    print("\n" + "="*50)
    print("🔍 DEBUG CV DATA")
    print("="*50)
    
    debug_user_directory()
    
    print("\n" + "-"*30)
    print("Loading CVs from session...")
    cvs = load_uploaded_cvs_from_session()
    
    print(f"\n✅ Final result: {len(cvs)} CVs loaded")
    
    return f"Debug completed. Check terminal for details. Loaded {len(cvs)} CVs."


def extract_cv_data(filename):
    """Extract information from CV file"""
    try:
        # Use user-specific file path
        file_path = get_user_file_path(filename)
        
        # Extract text from file
        text = extract_text(file_path)
        if not text:
            return None

        # Extract entities
        entities = extract_entities_ner(text)
        emails, phones, years_exp = extract_basic_entities(text)
        skills = extract_skills_spacy(text)

        # Create CV data
        cv_data = {
            'filename': filename,
            'file_path': file_path,  # User-specific path
            'name': entities['PERSON'][0] if entities['PERSON'] else filename,
            'email': emails[0] if emails else "N/A",
            'phone': phones[0] if phones else "N/A", 
            'years_exp': years_exp,
            'skills': skills,
            'skills_text': ", ".join(skills[:10]),
            'text': text,  # Save full text for search
            'entities': entities,
            'emails': emails,
            'phones': phones
        }

        return cv_data
        
    except Exception as e:
        print(f"Error extracting CV data from {filename}: {e}")
        return None


