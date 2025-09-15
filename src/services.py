import os
import re
import json
from datetime import datetime
from typing import List, Tuple

import spacy
import PyPDF2
import docx
import numpy as np
from flask import session
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# Load spaCy model and sentence-transformers once
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

skills_master: List[str] = []


def load_skills() -> None:
    global skills_master
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        skills_master = [s.strip().lower() for s in data.get("technical_skills", []) if s.strip()]
    except FileNotFoundError:
        skills_master = [
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
            "sql", "mysql", "postgresql", "mongodb", "redis",
            "html", "css", "react", "angular", "vue", "node", "express", "next.js", "nest.js",
            "django", "flask", "fastapi", "spring", "spring boot", "dotnet", ".net",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
            "linux", "git", "ci/cd", "jenkins", "github actions",
            "machine learning", "deep learning", "nlp", "computer vision",
            "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
            "big data", "hadoop", "spark", "kafka",
            "rest", "graphql", "microservices", "oop", "design patterns",
            "agile", "scrum", "jira"
        ]


def get_skill_matcher() -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_master]
    matcher.add("SKILL", patterns)
    return matcher


# Initialize skills and skill_matcher
load_skills()
skill_matcher = get_skill_matcher()


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = ""
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text
    if ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def extract_entities_ner(text: str) -> dict:
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": [], "LOC": [], "TITLE": []}
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
            entities[ent.label_].append(ent.text)

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name_found = False
    for line in lines[:5]:
        if line.lower().startswith(('name:', 'candidate')):
            name = line.split(':', 1)[1].strip() if ':' in line else line
            if name:
                entities["PERSON"].insert(0, name)
                name_found = True
                break
    if not name_found and lines:
        first_line = lines[0]
        if not (first_line.startswith(('Email:', 'Phone:', 'Current Location:', 'Visa Status:')) or 
                '@' in first_line or 
                re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', first_line)):
            entities["PERSON"].insert(0, first_line)

    titles = []
    title_keywords = [
        "engineer", "developer", "manager", "scientist", "analyst",
        "lead", "architect", "specialist", "consultant", "director",
        "senior", "junior", "principal", "staff", "head", "sr.", "sr"
    ]
    words = text.lower().split()
    for i, word in enumerate(words):
        if any(keyword in word for keyword in title_keywords):
            start = max(0, i - 2)
            end = min(len(words), i + 3)
            title_phrase = " ".join(words[start:end])
            titles.append(title_phrase.title())
    entities["TITLE"] = list(dict.fromkeys(titles))[:5]

    for key in entities:
        entities[key] = list(dict.fromkeys([v.strip() for v in entities[key] if v.strip()]))[:10]

    return entities


def extract_skills_spacy(text: str) -> List[str]:
    """Extract skills from text using spaCy phrase matcher"""
    try:
        doc = nlp(text)
        spans = [doc[start:end].text for _, start, end in skill_matcher(doc)]
        unique_skills = list(dict.fromkeys([s.strip().lower() for s in spans if s.strip()]))
        
        # If no skills found with matcher, try fallback
        if not unique_skills:
            print("No skills found with matcher, trying fallback...")
            # Find common technical keywords in text
            tech_keywords = [
                "java", "python", "javascript", "typescript", "c++", "c#", "sql", "html", "css",
                "react", "angular", "vue", "node", "spring", "hibernate", "mysql", "oracle",
                "aws", "azure", "docker", "kubernetes", "git", "jenkins", "maven", "gradle",
                "rest", "soap", "json", "xml", "junit", "agile", "scrum", "linux", "windows"
            ]
            
            text_lower = text.lower()
            found_skills = []
            for keyword in tech_keywords:
                if keyword in text_lower:
                    found_skills.append(keyword)
            
            unique_skills = list(dict.fromkeys(found_skills))
            print(f"Fallback found {len(unique_skills)} skills: {unique_skills}")
        
        return unique_skills[:15]  # Limit to maximum 15 skills
        
    except Exception as e:
        print(f"Error in extract_skills_spacy: {e}")
        return []


def extract_basic_entities(text: str) -> Tuple[List[str], List[str], float]:
    """Extract email, phone and years of experience from text with extended patterns"""
    emails = re.findall(r'\S+@\S+', text)
    
    # Extract phone numbers
    phone_pattern = r'\d{3}[-.]?\d{3}[-.]?\d{4}'
    phone_matches = re.findall(phone_pattern, text)
    phones = []
    for phone in phone_matches:
        digits_only = re.sub(r'[^\d]', '', phone)
        if len(digits_only) == 10:
            formatted = f"{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"
            phones.append(formatted)
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            formatted = f"{digits_only[1:4]}-{digits_only[4:7]}-{digits_only[7:]}"
            phones.append(formatted)
        else:
            phones.append(phone)
    phones = list(dict.fromkeys(phones))
    
    # Extract years of experience with extended patterns
    years_patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years|yrs|yr)\s*(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp)[\s\w]*?(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years|yrs|yr)',
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years|yrs|yr)',
        r'(?:over|more\s*than|at\s*least)\s*(\d+(?:\.\d+)?)\s*(?:year|years|yrs|yr)',
        r'(?:up\s*to|maximum|max)\s*(\d+(?:\.\d+)?)\s*(?:year|years|yrs|yr)'
    ]
    
    years_exp = 0.0
    for pattern in years_patterns:
        years_matches = re.findall(pattern, text.lower())
        if years_matches:
            # Get the highest number of years found
            years_exp = max([float(y) for y in years_matches])
            break
    
    return emails, phones, years_exp


def calculate_semantic_similarity(resume_texts: List[str], job_description: str) -> List[float]:
    all_texts = resume_texts + [job_description]
    embeddings = sbert_model.encode(all_texts, normalize_embeddings=True)
    job_embedding = embeddings[-1]
    resume_embeddings = embeddings[:-1]
    similarities: List[float] = []
    for resume_emb in resume_embeddings:
        similarity = float(np.dot(resume_emb, job_embedding))
        similarities.append(similarity)
    return similarities


def save_cv_to_session(cv_data: dict) -> None:
    """Save CV data to session and JSON file in user directory"""
    try:
        if 'uploaded_cvs' not in session:
            session['uploaded_cvs'] = []
        
        # ✅ Use user directory for JSON file
        user_dir = get_user_uploads_dir()
        json_file_path = os.path.join(user_dir, f"{cv_data['filename']}.json")
        print(f"Saving CV data to user directory: {json_file_path}")
        
        minimal_cv_data = {
            'filename': cv_data['filename'],
            'name': cv_data.get('name', 'Unknown'),
            'email': cv_data.get('email', 'N/A'),
            'phone': cv_data.get('phone', 'N/A'),
            'years_exp': cv_data.get('years_exp', 0),
            'skills': cv_data.get('skills', []),
            'skills_text': cv_data.get('skills_text', ''),
            'text': cv_data.get('text', ''),
            'entities': cv_data.get('entities', {}),
            'emails': cv_data.get('emails', []),
            'phones': cv_data.get('phones', []),
            'file_path': cv_data.get('file_path', ''),
            'feedback_history': [],
            'latest_feedback': None
        }
        
        # Ensure user directory exists
        os.makedirs(user_dir, exist_ok=True)
        
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(minimal_cv_data, json_file, ensure_ascii=False, indent=4)
        
        print(f"✅ Saved JSON file in user directory: {json_file_path}")
        
        # Check if file exists
        if os.path.exists(json_file_path):
            print(f"✅ JSON file created successfully: {json_file_path}")
        else:
            print(f"❌ Error: JSON file not created: {json_file_path}")
        
        session['uploaded_cvs'].append({'json_file_path': json_file_path})
        session.modified = True
        
        print(f"✅ Session updated. Total CVs in session: {len(session['uploaded_cvs'])}")
        
    except Exception as e:
        print(f"❌ Error saving CV data: {e}")


def load_uploaded_cvs_from_session() -> list:
    """Load CV data from session and JSON files in user directory"""
    cvs = []
    print(f"Loading CVs from session. Session has {len(session.get('uploaded_cvs', []))} CVs")
    
    # ✅ Always try auto-reload if session is empty or has issues
    if not session.get('uploaded_cvs'):
        print("⚠️ Session empty, trying auto-reload from user directory...")
        auto_reload_cvs_from_user_directory()
    
    for rec in session.get('uploaded_cvs', []):
        json_file_path = rec.get('json_file_path')
        print(f"Checking JSON file: {json_file_path}")
        
        if json_file_path and os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as jf:
                    cv_data = json.load(jf)
                    cvs.append(cv_data)
                    print(f"✅ Loaded CV: {cv_data.get('filename', 'Unknown')}")
            except Exception as e:
                print(f"❌ Error reading JSON file {json_file_path}: {e}")
        else:
            print(f"⚠️ JSON file does not exist: {json_file_path}")
    
    print(f"✅ Total CVs loaded: {len(cvs)}")
    return cvs

def auto_reload_cvs_from_user_directory():
    """Auto-reload CVs from user directory when session is empty"""
    try:
        user_dir = get_user_uploads_dir()
        print(f"🔍 Scanning user directory: {user_dir}")
        
        if not os.path.exists(user_dir):
            print(f"❌ User directory does not exist: {user_dir}")
            return
        
        # Find all JSON files in user directory
        json_files = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(user_dir, filename)
                json_files.append(json_path)
        
        print(f"📁 Found {len(json_files)} JSON files in user directory")
        
        if json_files:
            # Initialize session if not exists
            if 'uploaded_cvs' not in session:
                session['uploaded_cvs'] = []
            
            # Reload from JSON files
            for json_path in json_files:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        cv_data = json.load(f)
                    
                    # Check if already exists in session
                    already_exists = False
                    for existing_cv in session['uploaded_cvs']:
                        if existing_cv.get('json_file_path') == json_path:
                            already_exists = True
                            break
                    
                    if not already_exists:
                        session['uploaded_cvs'].append({'json_file_path': json_path})
                        print(f"✅ Auto-reloaded CV: {cv_data.get('filename', 'Unknown')}")
                
                except Exception as e:
                    print(f"❌ Error loading JSON file {json_path}: {e}")
            
            session.modified = True
            print(f"✅ Auto-reload completed. Session now has {len(session['uploaded_cvs'])} CVs")
        else:
            print("📁 No JSON files found in user directory")
            
    except Exception as e:
        print(f"❌ Error in auto_reload_cvs_from_user_directory: {e}")


def dedupe_cvs(cvs_list=None) -> list:
    """Remove duplicate CVs and reload from user directory"""
    if cvs_list is None:
        # If no parameter, use session
        if 'uploaded_cvs' not in session:
            return []
        cvs_list = session['uploaded_cvs']
    
    print(f"Total CVs in list: {len(cvs_list)}")
    
    # Get all file paths from CV list
    file_paths = []
    for cv in cvs_list:
        json_file_path = cv.get('json_file_path')
        if json_file_path and os.path.exists(json_file_path):
            file_paths.append(json_file_path)
    
    # Clear old session if using session
    if cvs_list is session.get('uploaded_cvs'):
        session['uploaded_cvs'] = []
    
    # Reload from user directory
    unique_cvs = []
    seen_files = set()
    
    for file_path in file_paths:
        if file_path in seen_files:
            continue
        seen_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
                unique_cvs.append(cv_data)
        except Exception as e:
            print(f"❌ Error loading CV from {file_path}: {e}")
            continue
    
    # Update session if using session
    if cvs_list is session.get('uploaded_cvs'):
        session['uploaded_cvs'] = unique_cvs
    
    print(f"✅ Unique CVs after deduplication: {len(unique_cvs)}")
    return unique_cvs


def debug_cv_data():
    """Debug function to check actual CV data"""
    if 'uploaded_cvs' not in session:
        print("No CVs in session")
        return []
    
    print(f"Total CVs in session: {len(session['uploaded_cvs'])}")
    
    for i, cv in enumerate(session['uploaded_cvs']):
        json_file_path = cv.get('json_file_path')
        if json_file_path and os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    cv_data = json.load(json_file)
                print(f"\nCV {i+1}:")
                print(f"  - Filename: {cv_data.get('filename', 'N/A')}")
                print(f"  - Name: {cv_data.get('name', 'N/A')}")
                print(f"  - Years Exp: {cv_data.get('years_exp', 'N/A')}")
                print(f"  - Skills: {cv_data.get('skills_text', 'N/A')[:100]}...")
                print(f"  - Text length: {len(cv_data.get('text', ''))}")
                if cv_data.get('text'):
                    text_sample = cv_data.get('text', '')[:200].lower()
                    print(f"  - Text sample: {text_sample}")
            except Exception as e:
                print(f"Error reading CV {i+1}: {e}")
    
    return session['uploaded_cvs']


def fix_cv_data_structure():
    """Fix CV data structure if corrupted"""
    if 'uploaded_cvs' not in session:
        return
    
    print("Checking and fixing CV data structure...")
    
    # Get all file paths
    file_paths = []
    for cv in session['uploaded_cvs']:
        json_file_path = cv.get('json_file_path')
        if json_file_path and os.path.exists(json_file_path):
            file_paths.append(json_file_path)
    
    # Clear old session
    session['uploaded_cvs'] = []
    
    # Recreate CV data from original files
    for file_path in file_paths:
        try:
            # Read original file (PDF/DOCX)
            original_file = file_path.replace('.json', '')
            if os.path.exists(original_file):
                # Re-extract data
                from control.index_controller import extract_cv_data
                filename = os.path.basename(original_file)
                cv_data = extract_cv_data(filename, original_file)
                
                if cv_data:
                    # Save with correct structure
                    save_cv_to_session(cv_data)
                    print(f"✓ Fixed CV: {filename}")
                else:
                    print(f"✗ Cannot fix CV: {filename}")
        except Exception as e:
            print(f"Error fixing CV {file_path}: {e}")
    
    session.modified = True
    print(f"Completed! Total CVs: {len(session['uploaded_cvs'])}")


def search_cvs_in_session(search_query: str, search_type: str = "all") -> list:
    """Search CVs in session with advanced search capabilities"""
    if 'uploaded_cvs' not in session:
        print("No CVs in session")
        return []
    
    # If session is empty, try auto-reload
    if not session.get('uploaded_cvs'):
        print("⚠️ Session empty, trying auto-reload...")
        auto_reload_cvs_from_user_directory()
        
        # If still empty after auto-reload
        if not session.get('uploaded_cvs'):
            print("❌ Still no CVs after auto-reload")
        return []
    
    results = []
    query_lower = search_query.lower()
    
    print(f"🔍 Searching in {len(session['uploaded_cvs'])} CVs...")
    
    for rec in session['uploaded_cvs']:
        json_file_path = rec.get('json_file_path')
        if json_file_path and os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    cv_data = json.load(json_file)
                
                # Search by search_type
                if search_type == "all" or search_type == "name":
                    if query_lower in cv_data.get('name', '').lower():
                        results.append(cv_data)
                        continue
                
                if search_type == "all" or search_type == "skills":
                    skills_text = ' '.join(cv_data.get('skills', [])).lower()
                    if query_lower in skills_text:
                        results.append(cv_data)
                        continue
                
                if search_type == "all" or search_type == "experience":
                    exp_text = str(cv_data.get('years_exp', 0))
                    if query_lower in exp_text:
                        results.append(cv_data)
                        continue
                
                if search_type == "all" or search_type == "education":
                    education_text = ' '.join(cv_data.get('education', [])).lower()
                    if query_lower in education_text:
                        results.append(cv_data)
                        continue
                    
            except Exception as e:
                print(f"Error reading CV file {json_file_path}: {e}")
                continue
    
    print(f"✅ Found {len(results)} matching CVs")
    return results


# Simple in-memory result store for ranking page
ranking_results: List[dict] = []


def set_ranking_results(res: List[dict]) -> None:
    global ranking_results
    ranking_results = res or []


def get_ranking_results() -> List[dict]:
    return ranking_results


def delete_cv_from_session(filename: str) -> bool:
    """Delete CV from session and file system in user directory"""
    try:
        print(f"🔍 Attempting to delete CV: {filename}")
        
        # ✅ Auto-reload CVs from user directory if session is empty
        if 'uploaded_cvs' not in session or not session.get('uploaded_cvs'):
            print("⚠️ Session empty, auto-reloading from user directory...")
            auto_reload_cvs_from_user_directory()
        
        if 'uploaded_cvs' not in session:
            print("❌ No uploaded_cvs in session after auto-reload")
            return False
        
        print(f"📋 Session has {len(session['uploaded_cvs'])} CVs")
        
        # Find and delete CV from session
        cv_to_remove = None
        for i, cv in enumerate(session['uploaded_cvs']):
            json_file_path = cv.get('json_file_path')
            print(f"  CV {i+1}: {json_file_path}")
            
            if json_file_path and os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r", encoding="utf-8") as json_file:
                        cv_data = json.load(json_file)
                    print(f"    Filename in JSON: {cv_data.get('filename')}")
                    print(f"    Looking for: {filename}")
                    
                    if cv_data.get('filename') == filename:
                        cv_to_remove = cv
                        print(f"    ✅ Found matching CV!")
                        break
                except Exception as e:
                    print(f"    ❌ Error reading JSON: {e}")
                    continue
            else:
                print(f"    ❌ JSON file does not exist: {json_file_path}")
        
        if cv_to_remove:
            # Delete JSON file from user directory
            json_file_path = cv_to_remove.get('json_file_path')
            if json_file_path and os.path.exists(json_file_path):
                os.remove(json_file_path)
                print(f"✅ Deleted JSON file from user directory: {json_file_path}")
            else:
                print(f"⚠️ JSON file does not exist: {json_file_path}")
            
            # Delete original file (PDF/DOCX) from user directory
            original_file = get_user_file_path(filename)
            if os.path.exists(original_file):
                os.remove(original_file)
                print(f"✅ Deleted original file from user directory: {original_file}")
            else:
                print(f"⚠️ Original file does not exist: {original_file}")
            
            # Remove from session
            session['uploaded_cvs'].remove(cv_to_remove)
            session.modified = True
            
            print(f"✅ Deleted CV: {filename}")
            return True
        else:
            print(f"❌ CV not found: {filename}")
            return False
            
    except Exception as e:
        print(f"❌ Error deleting CV {filename}: {e}")
        return False


def save_jd_to_session(jd_data: dict) -> None:
    """Save JD data to session"""
    try:
        session['current_jd'] = {
            'filename': jd_data.get('filename', ''),
            'content': jd_data.get('content', ''),
            'upload_time': datetime.now().isoformat(),
            'file_size': jd_data.get('file_size', 0)
        }
        session.modified = True
        print(f"Saved JD to session: {jd_data.get('filename', 'Unknown')}")
    except Exception as e:
        print(f"Error saving JD to session: {e}")


def get_jd_from_session() -> dict:
    """Get JD data from session"""
    return session.get('current_jd', {})


def clear_jd_from_session() -> None:
    """Clear JD from session"""
    if 'current_jd' in session:
        del session['current_jd']
        session.modified = True
        print("Cleared JD from session")


# ===== USER DIRECTORY MANAGEMENT =====

def get_user_uploads_dir():
    """Get uploads directory for current user"""
    user_id = session.get('user_id', 'anonymous')
    user_dir = os.path.join("uploads", f"user_{user_id}")
    
    # Create directory if not exists
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)
        print(f"✅ Created user directory: {user_dir}")
    
    return user_dir


def clear_user_uploads_directory():
    """Delete all files in current user's uploads directory"""
    try:
        user_dir = get_user_uploads_dir()
        
        if not os.path.exists(user_dir):
            print(f"User directory {user_dir} does not exist")
            return True
        
        deleted_count = 0
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted file: {filename}")
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                    deleted_count += 1
                    print(f"Deleted directory: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        
        print(f"✅ Deleted {deleted_count} files/directories from {user_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting user uploads directory: {e}")
        return False


def get_user_file_path(filename):
    """Get file path in user directory"""
    user_dir = get_user_uploads_dir()
    return os.path.join(user_dir, filename)


def save_file_to_user_dir(file, filename):
    """Save file to user directory"""
    try:
        user_dir = get_user_uploads_dir()
        file_path = os.path.join(user_dir, filename)
        file.save(file_path)
        print(f"✅ Saved file to user directory: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Error saving file {filename}: {e}")
        return None


def list_user_files():
    """List all files of current user"""
    try:
        user_dir = get_user_uploads_dir()
        if not os.path.exists(user_dir):
            return []
        
        files = []
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            if os.path.isfile(file_path):
                file_info = {
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'created': datetime.fromtimestamp(os.path.getctime(file_path))
                }
                files.append(file_info)
        
        return files
    except Exception as e:
        print(f"❌ Error listing user files: {e}")
        return []


def cleanup_old_user_sessions(days_old=7):
    """Clean up old user directories"""
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            return True
        
        current_time = datetime.now()
        cleaned_count = 0
        
        for user_folder in os.listdir(uploads_dir):
            user_path = os.path.join(uploads_dir, user_folder)
            if not os.path.isdir(user_path) or not user_folder.startswith('user_'):
                continue
            
            # Check folder creation time
            folder_time = datetime.fromtimestamp(os.path.getctime(user_path))
            if (current_time - folder_time).days > days_old:
                import shutil
                shutil.rmtree(user_path)
                cleaned_count += 1
                print(f"🗑️ Deleted old user folder: {user_path}")
        
        print(f"✅ Cleaned up {cleaned_count} old user folders")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning up old user folders: {e}")
        return False


def clear_uploads_directory():
    """Delete files in current user's uploads directory (does not affect other users)"""
    return clear_user_uploads_directory()


def clear_user_data_only():
    """Delete user data but keep login session"""
    try:
        # Delete CV data
        if 'uploaded_cvs' in session:
            del session['uploaded_cvs']
        
        # Delete JD data
        if 'current_jd' in session:
            del session['current_jd']
        
        # Delete ranking results
        if 'ranking_results' in session:
            del session['ranking_results']
        
        # DO NOT delete user_id and logged_in
        session.modified = True
        
        print("Deleted user data (kept login session)")
        return True
        
    except Exception as e:
        print(f"Error deleting user data: {e}")
        return False


def clear_session_data():
    """Delete all session data"""
    try:
        # Delete CV data
        if 'uploaded_cvs' in session:
            del session['uploaded_cvs']
        
        # Delete JD data
        if 'current_jd' in session:
            del session['current_jd']
        
        # Delete ranking results
        if 'ranking_results' in session:
            del session['ranking_results']
        
        # Clear entire session
        session.clear()
        session.modified = True
        
        print("Deleted all session data")
        return True
        
    except Exception as e:
        print(f"Error deleting session data: {e}")
        return False


def debug_user_directory():
    """Debug function to check user directory and files"""
    try:
        user_dir = get_user_uploads_dir()
        print(f"\n🔍 DEBUG: User Directory Analysis")
        print(f"User ID: {session.get('user_id', 'anonymous')}")
        print(f"User Directory: {user_dir}")
        print(f"Directory exists: {os.path.exists(user_dir)}")
        
        if os.path.exists(user_dir):
            files = os.listdir(user_dir)
            print(f"Files in directory: {len(files)}")
            
            json_files = [f for f in files if f.endswith('.json')]
            print(f"JSON files: {len(json_files)}")
            
            for json_file in json_files:
                json_path = os.path.join(user_dir, json_file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        cv_data = json.load(f)
                    print(f"  ✅ {json_file}: {cv_data.get('filename', 'Unknown')}")
                except Exception as e:
                    print(f"  ❌ {json_file}: Error - {e}")
        
        print(f"Session uploaded_cvs: {len(session.get('uploaded_cvs', []))}")
        for i, cv in enumerate(session.get('uploaded_cvs', [])):
            print(f"  Session CV {i+1}: {cv.get('json_file_path', 'No path')}")
        
    except Exception as e:
        print(f"❌ Error in debug_user_directory: {e}")


