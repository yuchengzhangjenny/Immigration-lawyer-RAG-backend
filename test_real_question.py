#!/usr/bin/env python3
"""
Test script for the Immigration Lawyer RAG Backend
Tests the /search endpoint with real immigration law questions
"""

import requests
import json
import time

# Backend URL
BASE_URL = "http://127.0.0.1:8081"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        print(f"\n🔍 Testing: {method} {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    print("🚀 Testing Immigration Lawyer RAG Backend")
    print("=" * 50)
    
    # Test 1: Health check
    test_endpoint("/api/health")
    
    # Test 2: API info
    test_endpoint("/")
    
    # Test 3: Real immigration questions
    questions = [
        {
            "query": "What are the requirements for H1B visa?",
            "use_llm": False
        },
        {
            "query": "How long can I stay in the US with a tourist visa?", 
            "use_llm": False
        },
        {
            "query": "What documents are needed for naturalization?",
            "use_llm": False
        }
    ]
    
    for i, question_data in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question_data['query']}")
        print("-" * 60)
        
        start_time = time.time()
        result = test_endpoint("/search", "POST", question_data)
        end_time = time.time()
        
        if result:
            print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
            
            # Display the answer nicely
            if 'answer' in result:
                print(f"\n💡 Answer:")
                print(result['answer'])
            
            if 'sources' in result:
                print(f"\n📚 Sources found: {len(result['sources'])}")
                for j, source in enumerate(result['sources'][:2], 1):  # Show first 2 sources
                    print(f"   {j}. {source.get('text', '')[:100]}...")
        
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    main() 