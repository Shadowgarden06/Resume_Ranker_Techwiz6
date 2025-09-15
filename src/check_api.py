#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Resume Ranker - API Security Check
Check API keys and allow application to run
"""

import os
import sys
import hashlib

def check_api_keys():
    """Check API keys"""
    print("🔍 Checking API keys...")
    
    # Get API keys from environment
    gemini_key = os.environ.get('GEMINI_API_KEY')
    security_key = os.environ.get('SECURITY_KEY')
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found!")
        return False
        
    if not security_key:
        print("❌ SECURITY_KEY not found!")
        return False
    
    # Check basic format
    if not gemini_key.startswith('AIzaSy'):
        print("❌ Invalid GEMINI_API_KEY format!")
        return False
        
    if not security_key.startswith('SK'):
        print("❌ Invalid SECURITY_KEY format!")
        return False
    
    # Check length
    if len(gemini_key) < 30:
        print("❌ GEMINI_API_KEY too short!")
        return False
        
    if len(security_key) < 20:
        print("❌ SECURITY_KEY too short!")
        return False
    
    print("✅ API keys format valid")
    return True

def check_system():
    """Check system requirements"""
    print("🖥️  Checking system...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required!")
        return False
    
    # Check required modules
    required = ['flask', 'spacy', 'pandas']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing modules: {', '.join(missing)}")
        print("Run: pip install flask spacy pandas")
        return False
    
    print("✅ System check passed")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("🛡️  AI Resume Ranker - Security Check")
    print("=" * 50)
    
    try:
        # Check API keys
        if not check_api_keys():
            return False
        
        # Check system
        if not check_system():
            return False
        
        print("=" * 50)
        print("✅ All checks passed!")
        print("🚀 Application authorized to run")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
