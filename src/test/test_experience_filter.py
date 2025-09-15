#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check experience filtering logic
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from control.search_controller import filter_by_experience

def test_experience_filter():
    """Test experience filtering logic"""
    print("🧪 Testing Experience Filter Logic")
    print("=" * 50)
    
    # Create test data
    test_cvs = [
        {'name': 'John Doe', 'years_exp': 2.0},
        {'name': 'Jane Smith', 'years_exp': 5.0},
        {'name': 'Bob Johnson', 'years_exp': 8.0},
        {'name': 'Alice Brown', 'years_exp': 12.0},
        {'name': 'Charlie Wilson', 'years_exp': 15.0},
        {'name': 'Diana Lee', 'years_exp': 0.0},
        {'name': 'Eve Davis', 'years_exp': 25.0},
    ]
    
    print(f"📋 Test CVs:")
    for cv in test_cvs:
        print(f"  - {cv['name']}: {cv['years_exp']} years")
    
    # Test cases
    test_cases = [
        {'min_exp': 5, 'max_exp': 20, 'description': 'Min 5 years, Max 20 years'},
        {'min_exp': 0, 'max_exp': 10, 'description': 'Min 0 years, Max 10 years'},
        {'min_exp': 10, 'max_exp': 999, 'description': 'Min 10 years, No max limit'},
        {'min_exp': 5, 'max_exp': 5, 'description': 'Exactly 5 years only'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"   Min: {test_case['min_exp']}, Max: {test_case['max_exp']}")
        
        filtered_cvs = filter_by_experience(test_cvs, test_case['min_exp'], test_case['max_exp'])
        
        print(f"   📊 Results ({len(filtered_cvs)}/{len(test_cvs)} CVs):")
        for cv in filtered_cvs:
            print(f"     ✅ {cv['name']}: {cv['years_exp']} years")
        
        # Show excluded CVs
        excluded_names = [cv['name'] for cv in test_cvs if cv not in filtered_cvs]
        if excluded_names:
            print(f"   ❌ Excluded: {', '.join(excluded_names)}")

if __name__ == "__main__":
    test_experience_filter()
