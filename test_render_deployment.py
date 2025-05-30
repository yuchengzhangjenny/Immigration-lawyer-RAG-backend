#!/usr/bin/env python3
"""
Test script for Immigration Lawyer RAG Backend deployed on Render.
Run this script to test all API endpoints.
"""

import requests
import json
import time

BASE_URL = "https://immigration-lawyer-rag-backend.onrender.com"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_search_endpoint(query, use_llm=False):
    """Test the search endpoint with a query."""
    print(f"\n🔍 Testing search with query: '{query}' (LLM: {use_llm})")
    try:
        payload = {
            "query": query,
            "use_llm": use_llm
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/search", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for AI processing
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Query: {result.get('query', 'N/A')}")
            print(f"Answer: {result.get('answer', 'N/A')[:200]}...")
            print(f"Retrieved Chunks: {len(result.get('retrieved_chunks', []))}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_frontend():
    """Test if the frontend is accessible."""
    print("\n🔍 Testing frontend accessibility...")
    try:
        response = requests.get(BASE_URL, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)} characters")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Immigration Lawyer RAG Backend on Render")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What are the requirements for H1B visa?",
        "How do I apply for asylum in the United States?",
        "What documents are needed for naturalization?",
        "What are the eligibility criteria for DACA?"
    ]
    
    # Run tests
    health_ok = test_health_check()
    frontend_ok = test_frontend()
    
    # Test search endpoints
    search_results = []
    for query in test_queries:
        # Test without LLM first (faster)
        result = test_search_endpoint(query, use_llm=False)
        search_results.append(result)
        
        # Test one query with LLM
        if query == test_queries[0]:
            llm_result = test_search_endpoint(query, use_llm=True)
            search_results.append(llm_result)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Frontend: {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    print(f"Search API: {sum(search_results)}/{len(search_results)} tests passed")
    
    if all([health_ok, frontend_ok]) and sum(search_results) > 0:
        print("\n🎉 Your deployment is working!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 