# src/download_helpers.py
import os
import json
import csv
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime
from flask import Response, jsonify
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def get_cv_feedback_info(filename):
    """
    Get feedback information from CV JSON file in user directory
    """
    try:
        # ✅ Use user directory for JSON file
        from services import get_user_file_path
        cv_json_file = get_user_file_path(f"{filename}.json")
        
        if not os.path.exists(cv_json_file):
            return {
                'total_feedback': 0,
                'latest_decision': None,
                'average_rating': None,
                'latest_feedback_time': None
            }
        
        with open(cv_json_file, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        
        feedback_history = cv_data.get('feedback_history', [])
        latest_feedback = cv_data.get('latest_feedback', None)
        
        # Calculate average rating
        ratings = [float(fb.get('human_rating', 0)) for fb in feedback_history 
                  if fb.get('human_rating') and str(fb.get('human_rating')).strip() != '']
        
        return {
            'total_feedback': len(feedback_history),
            'latest_decision': latest_feedback.get('decision') if latest_feedback else None,
            'average_rating': round(sum(ratings) / len(ratings), 2) if ratings else None,
            'latest_feedback_time': latest_feedback.get('timestamp') if latest_feedback else None
        }
        
    except Exception as e:
        print(f"❌ Error getting CV feedback: {e}")
        return {
            'total_feedback': 0,
            'latest_decision': None,
            'average_rating': None,
            'latest_feedback_time': None
        }

def escape_csv_value(value):
    """Escape CSV values to avoid format errors"""
    if value is None:
        return ""
    str_value = str(value)
    if '"' in str_value:
        str_value = str_value.replace('"', '""')
    if ',' in str_value or '"' in str_value or '\n' in str_value:
        return f'"{str_value}"'
    return str_value

def create_csv_response(data, filename_prefix):
    """Create common CSV response"""
    try:
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = [
            'CV Name', 'Full Name', 'Email', 'Phone Number', 
            'Experience (years)', 'Skills', 'Total Feedback',
            'Latest Decision', 'Average Rating', 'Latest Feedback Time'
        ]
        writer.writerow(headers)
        
        # Data
        for item in data:
            feedback_info = get_cv_feedback_info(item.get('filename', 'Unknown'))
            
            row = [
                escape_csv_value(item.get('filename', 'N/A')),
                escape_csv_value(item.get('name', 'N/A')),
                escape_csv_value(item.get('email', 'N/A')),
                escape_csv_value(item.get('phone', 'N/A')),
                item.get('years_exp', 0),
                escape_csv_value(item.get('skills_text', 'N/A')),
                feedback_info['total_feedback'],
                escape_csv_value(feedback_info['latest_decision']),
                feedback_info['average_rating'],
                escape_csv_value(feedback_info['latest_feedback_time'])
            ]
            writer.writerow(row)
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error creating CSV: {e}")
        return jsonify({'success': False, 'message': f'Error creating CSV: {str(e)}'}), 500

def create_excel_response(data, filename_prefix):
    """Create common Excel response"""
    try:
        # Prepare data
        excel_data = []
        for item in data:
            feedback_info = get_cv_feedback_info(item.get('filename', 'Unknown'))
            
            excel_data.append({
                'CV Name': item.get('filename', 'N/A'),
                'Full Name': item.get('name', 'N/A'),
                'Email': item.get('email', 'N/A'),
                'Phone Number': item.get('phone', 'N/A'),
                'Experience (years)': item.get('years_exp', 0),
                'Skills': item.get('skills_text', 'N/A'),
                'Total Feedback': feedback_info['total_feedback'],
                'Latest Decision': feedback_info['latest_decision'],
                'Average Rating': feedback_info['average_rating'],
                'Latest Feedback Time': feedback_info['latest_feedback_time']
            })
        
        # Create Excel
        df = pd.DataFrame(excel_data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Format sheet
            worksheet = writer.sheets['Data']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error creating Excel: {e}")
        return jsonify({'success': False, 'message': f'Error creating Excel: {str(e)}'}), 500

def create_pdf_response(data, filename_prefix, title):
    """Create common PDF response (Landscape mode)"""
    try:
        output = BytesIO()
        
        # Use landscape mode
        from reportlab.lib.pagesizes import landscape, A4
        doc = SimpleDocTemplate(output, pagesize=landscape(A4),
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.3*inch, rightMargin=0.3*inch)
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=20,
            alignment=1
        )
        title_para = Paragraph(title, title_style)
        
        # Table data (optimized for landscape)
        table_data = [['CV Name', 'Full Name', 'Email', 'Phone', 'Experience', 'Skills', 'Feedback', 'Decision', 'Rating']]
        
        for item in data:
            feedback_info = get_cv_feedback_info(item.get('filename', 'Unknown'))
            
            feedback_text = f"Count: {feedback_info['total_feedback']}"
            decision_text = feedback_info['latest_decision'] or 'N/A'
            rating_text = str(feedback_info['average_rating']) if feedback_info['average_rating'] else 'N/A'
            
            # Truncate long text for landscape
            cv_name = item.get('filename', 'Unknown')[:20] + '...' if len(item.get('filename', '')) > 20 else item.get('filename', 'Unknown')
            full_name = item.get('name', 'N/A')[:25] + '...' if len(item.get('name', '')) > 25 else item.get('name', 'N/A')
            email = item.get('email', 'N/A')[:25] + '...' if len(item.get('email', '')) > 25 else item.get('email', 'N/A')
            phone = item.get('phone', 'N/A')[:15] + '...' if len(item.get('phone', '')) > 15 else item.get('phone', 'N/A')
            skills = item.get('skills_text', 'N/A')[:30] + '...' if len(item.get('skills_text', '')) > 30 else item.get('skills_text', 'N/A')
            
            table_data.append([
                cv_name,
                full_name,
                email,
                phone,
                f"{item.get('years_exp', 0)}y",
                skills,
                feedback_text,
                decision_text[:20] + '...' if len(decision_text) > 20 else decision_text,
                rating_text
            ])
        
        # Table with column widths optimized for landscape
        col_widths = [60, 80, 90, 60, 40, 100, 40, 60, 30]
        table = Table(table_data, colWidths=col_widths)
        
        # Table style
        table_style = TableStyle([
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
        ])
        
        table.setStyle(table_style)
        
        # Build PDF
        doc.build([title_para, table])
        
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pdf"
        
        return Response(
            output.getvalue(),
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
        
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return jsonify({'success': False, 'message': f'Error creating PDF: {str(e)}'}), 500
