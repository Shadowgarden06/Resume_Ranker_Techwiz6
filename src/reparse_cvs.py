#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to re-parse all existing CVs to extract skills properly
"""

import os
import json
import sys
from datetime import datetime

def get_user_uploads_dir():
    """Get user-specific uploads directory"""
    from services import get_user_uploads_dir
    return get_user_uploads_dir()

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file"""
    try:
        from services import extract_text
        return extract_text(file_path)
    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {e}")
        return None

def extract_entities_from_text(text):
    """Extract entities from text using NER"""
    try:
        from services import extract_entities_ner, extract_basic_entities, extract_skills_spacy
        return {
            'entities': extract_entities_ner(text),
            'basic': extract_basic_entities(text),
            'skills': extract_skills_spacy(text)
        }
    except Exception as e:
        print(f"❌ Error extracting entities: {e}")
        return None

def reparse_cv_file(cv_file_path):
    """Re-parse a single CV file"""
    print(f"📄 Processing: {os.path.basename(cv_file_path)}")
    
    # Extract text
    text = extract_text_from_file(cv_file_path)
    if not text:
        return False
    
    # Extract entities
    entities = extract_entities_from_text(text)
    if not entities:
        return False
    
    # Create CV data structure
    cv_data = {
        'filename': os.path.basename(cv_file_path),
        'text': text,
        'entities': entities['entities'],
        'name': entities['basic'].get('name', ''),
        'email': entities['basic'].get('email', ''),
        'phone': entities['basic'].get('phone', ''),
        'years_exp': entities['basic'].get('years_exp', 0),
        'skills': entities['skills'],
        'parsed_at': datetime.now().isoformat()
    }
    
    # Save JSON file
    json_file_path = cv_file_path + '.json'
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(cv_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {os.path.basename(json_file_path)}")
        return True
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🔄 AI Resume Ranker - CV Re-parsing Tool")
    print("=" * 60)
    
    # Get user directory
    user_dir = get_user_uploads_dir()
    if not os.path.exists(user_dir):
        print(f"❌ User directory not found: {user_dir}")
        return
    
    print(f"📁 Processing directory: {user_dir}")
    
    # Find all CV files
    cv_files = []
    for file in os.listdir(user_dir):
        if file.lower().endswith(('.pdf', '.docx', '.doc')):
            cv_files.append(os.path.join(user_dir, file))
    
    if not cv_files:
        print("❌ No CV files found!")
        return
    
    print(f"📊 Found {len(cv_files)} CV files")
    print()
    
    # Process each CV
    success_count = 0
    for cv_file in cv_files:
        if reparse_cv_file(cv_file):
            success_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print("📋 REPARSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(cv_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(cv_files) - success_count}")
    
    if success_count > 0:
        print("✅ Re-parsing completed successfully!")
    else:
        print("❌ Re-parsing failed!")

if __name__ == "__main__":
    main()
