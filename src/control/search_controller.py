# moved from src/search_controller.py
from flask import Blueprint, render_template, request, flash, jsonify, session
from auth import login_required
from services import search_cvs_in_session, load_uploaded_cvs_from_session
import re


bp = Blueprint('search_bp', __name__)


def filter_by_score(cvs, min_score, max_score):
    """Filter CVs by score"""
    try:
        min_score = float(min_score) if min_score else 0
        max_score = float(max_score) if max_score else 100
        filtered_cvs = []
        for cv in cvs:
            # Check if CV has combined_score
            score = cv.get('combined_score', 0)
            if isinstance(score, (int, float)) and min_score <= score <= max_score:
                filtered_cvs.append(cv)
        return filtered_cvs
    except Exception as e:
        print(f"Error filtering by score: {e}")
        return cvs


def filter_by_experience(cvs, min_exp, max_exp):
    """Filter CVs by years of experience"""
    try:
        min_exp = float(min_exp) if min_exp else 0
        max_exp = float(max_exp) if max_exp else 20
        filtered_cvs = []
        for cv in cvs:
            # Check if CV has years_exp
            exp = cv.get('years_exp', 0)
            if isinstance(exp, (int, float)) and min_exp <= exp <= max_exp:
                filtered_cvs.append(cv)
        return filtered_cvs
    except Exception as e:
        print(f"Error filtering by experience: {e}")
        return cvs


def filter_by_skills(cvs, required_skills):
    """Filter CVs by required skills"""
    if not required_skills:
        return cvs
    
    try:
        skills_list = [skill.strip().lower() for skill in required_skills.split(',') if skill.strip()]
        if not skills_list:
            return cvs
        
        filtered_cvs = []
        for cv in cvs:
            cv_skills = cv.get('skills_text', '').lower()
            if cv_skills and any(skill in cv_skills for skill in skills_list):
                filtered_cvs.append(cv)
        
        return filtered_cvs
    except Exception as e:
        print(f"Error filtering by skills: {e}")
        return cvs


def filter_by_education(cvs, education_level):
    """Filter CVs by education level"""
    if not education_level:
        return cvs
    
    try:
        education_keywords = {
            'high_school': ['high school', 'secondary school'],
            'bachelor': ['bachelor', 'university', 'college'],
            'master': ['master', 'mba', 'msc'],
            'phd': ['phd', 'doctor', 'doctoral'],
            'certification': ['certification', 'certificate', 'certified']
        }
        
        keywords = education_keywords.get(education_level, [])
        if not keywords:
            return cvs
        
        filtered_cvs = []
        for cv in cvs:
            cv_text = (cv.get('text', '') + ' ' + cv.get('skills_text', '')).lower()
            if cv_text and any(keyword in cv_text for keyword in keywords):
                filtered_cvs.append(cv)
        
        return filtered_cvs
    except Exception as e:
        print(f"Error filtering by education: {e}")
        return cvs


def sort_cvs(cvs, sort_by):
    """Sort CVs by criteria"""
    try:
        if sort_by == 'score':
            return sorted(cvs, key=lambda x: x.get('combined_score', 0), reverse=True)
        elif sort_by == 'experience':
            return sorted(cvs, key=lambda x: x.get('years_exp', 0), reverse=True)
        elif sort_by == 'name':
            return sorted(cvs, key=lambda x: x.get('name', '').lower())
        elif sort_by == 'skills':
            return sorted(cvs, key=lambda x: len(x.get('skills_text', '').split(',')), reverse=True)
        else:
            return cvs
    except Exception as e:
        print(f"Error sorting: {e}")
        return cvs


def prepare_cv_for_search(cv):
    """Prepare CV data for search"""
    try:
        # Ensure CV has all necessary fields
        prepared_cv = {
            'name': cv.get('name', 'Unknown'),
            'email': cv.get('email', 'N/A'),
            'phone': cv.get('phone', 'N/A'),
            'years_exp': cv.get('years_exp', 0),
            'skills_text': cv.get('skills_text', ''),
            'filename': cv.get('filename', ''),
            'text': cv.get('text', ''),
            'combined_score': cv.get('combined_score', 0),
            'tfidf_score': cv.get('tfidf_score', 0),
            'semantic_score': cv.get('semantic_score', 0),
            'skill_score': cv.get('skill_score', 0)
        }
        return prepared_cv
    except Exception as e:
        print(f"Error preparing CV: {e}")
        return cv


@bp.route('/search', methods=['GET', 'POST'])
@login_required
def search_candidates():
    try:
        if request.method == 'POST':
            # Get search parameters
            search_query = request.form.get('search_query', '').strip()
            search_type = request.form.get('search_type', 'all')
            
            # Get advanced filters
            min_score = request.form.get('min_score', '0')
            max_score = request.form.get('max_score', '100')
            min_experience = request.form.get('min_experience', '0')
            max_experience = request.form.get('max_experience', '20')
            required_skills = request.form.get('required_skills', '')
            education_level = request.form.get('education_level', '')
            sort_by = request.form.get('sort_by', 'score')
            
            # Load all CVs from session
            all_cvs = load_uploaded_cvs_from_session()
            
            if not all_cvs:
                return render_template('search_results.html', 
                                       search_query=search_query,
                                       search_type=search_type,
                                       results=[],
                                       message="No CVs in the system. Please upload CVs first.")
            
            # Prepare CV data
            prepared_cvs = [prepare_cv_for_search(cv) for cv in all_cvs]
            
            # Apply basic search
            if search_query:
                try:
                    search_results = search_cvs_in_session(search_query, search_type)
                    # Prepare search results
                    search_results = [prepare_cv_for_search(cv) for cv in search_results]
                except Exception as e:
                    print(f"Search error: {e}")
                    search_results = prepared_cvs
            else:
                search_results = prepared_cvs
            
            # Apply advanced filters
            filtered_results = search_results
            
            # Filter by score
            filtered_results = filter_by_score(filtered_results, min_score, max_score)
            
            # Filter by experience
            filtered_results = filter_by_experience(filtered_results, min_experience, max_experience)
            
            # Filter by skills
            filtered_results = filter_by_skills(filtered_results, required_skills)
            
            # Filter by education level
            filtered_results = filter_by_education(filtered_results, education_level)
            
            # Sort results
            filtered_results = sort_cvs(filtered_results, sort_by)
            
            # Create result message
            result_message = f"Found {len(filtered_results)} matching candidates"
            if search_query:
                result_message += f" with keyword '{search_query}'"
            
            # Save results to session for download
            session['last_search_results'] = filtered_results

            return render_template('search_results.html', 
                                   search_query=search_query,
                                   search_type=search_type,
                                   results=filtered_results,
                                   message=result_message,
                                   filters_applied={
                                       'min_score': min_score,
                                       'max_score': max_score,
                                       'min_experience': min_experience,
                                       'max_experience': max_experience,
                                       'required_skills': required_skills,
                                       'education_level': education_level,
                                       'sort_by': sort_by
                                   })
        
        return render_template('search.html')
        
    except Exception as e:
        print(f"Error in search_candidates: {e}")
        flash(f'Error occurred during search: {str(e)}', 'error')
        return render_template('search.html')


@bp.route('/debug_cvs')
@login_required
def debug_cvs():
    """Debug route to check CV data"""
    from services import debug_cv_data
    debug_cv_data()
    return "Debug info printed to console. Check server logs."


@bp.route('/fix_cv_data')
@login_required
def fix_cv_data():
    """Route to fix CV data structure"""
    from services import fix_cv_data_structure
    fix_cv_data_structure()
    return "CV data structure fixed. Check server logs."


@bp.route('/download_search_results_csv')
@login_required
def download_search_results_csv():
    """Download search results as CSV with feedback information"""
    try:
        from download_helpers import create_csv_response
        
        # Get data from session
        search_results = session.get('last_search_results', [])
        
        if not search_results:
            return jsonify({'success': False, 'message': 'No search results available for download'}), 404
        
        return create_csv_response(search_results, "search_results")
        
    except Exception as e:
        print(f"❌ Error in download_search_results_csv: {e}")
        return jsonify({'success': False, 'message': f'Error creating CSV: {str(e)}'}), 500


@bp.route('/download_search_results_excel')
@login_required  
def download_search_results_excel():
    """Download search results as Excel with feedback information"""
    try:
        from download_helpers import create_excel_response
        
        # Get data from session
        search_results = session.get('last_search_results', [])
        
        if not search_results:
            return jsonify({'success': False, 'message': 'No search results available for download'}), 404
        
        return create_excel_response(search_results, "search_results")
        
    except Exception as e:
        print(f"❌ Error in download_search_results_excel: {e}")
        return jsonify({'success': False, 'message': f'Error creating Excel: {str(e)}'}), 500


@bp.route('/download_search_results_pdf')
@login_required
def download_search_results_pdf():
    """Download search results as PDF with feedback information"""
    try:
        from download_helpers import create_pdf_response
        
        # Get data from session
        search_results = session.get('last_search_results', [])
        
        if not search_results:
            return jsonify({'success': False, 'message': 'No search results available for download'}), 404
        
        return create_pdf_response(search_results, "search_results", "CV Search Results with Feedback")
        
    except Exception as e:
        print(f"❌ Error in download_search_results_pdf: {e}")
        return jsonify({'success': False, 'message': f'Error creating PDF: {str(e)}'}), 500


