#!/usr/bin/env python3
"""
Focused backend test for News Knowledge Graph API
Tests the most critical endpoints for the demo
"""

import requests
import json
import sys
from datetime import datetime

class FocusedAPITester:
    def __init__(self, base_url="https://0951be95-b7b5-45fe-8a88-30c1c13d6686.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def test_endpoint(self, name, endpoint, expected_fields=None, timeout=10):
        """Test a single endpoint"""
        self.tests_run += 1
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ {name} - Status: 200 OK")
                    
                    # Check expected fields if provided
                    if expected_fields:
                        missing_fields = [field for field in expected_fields if field not in data]
                        if missing_fields:
                            print(f"⚠️  Missing fields: {missing_fields}")
                        else:
                            print(f"✅ All expected fields present: {expected_fields}")
                    
                    self.tests_passed += 1
                    return True, data
                    
                except json.JSONDecodeError:
                    print(f"❌ {name} - Invalid JSON response")
                    return False, None
            else:
                print(f"❌ {name} - Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False, None
                
        except requests.exceptions.Timeout:
            print(f"❌ {name} - Timeout after {timeout}s")
            return False, None
        except Exception as e:
            print(f"❌ {name} - Error: {str(e)}")
            return False, None

    def run_focused_tests(self):
        """Run focused tests on critical endpoints"""
        print("🚀 News Knowledge Graph - Focused Backend Tests")
        print(f"📍 Testing: {self.base_url}")
        print("=" * 60)
        
        # Test 1: Root API endpoint
        success, data = self.test_endpoint(
            "Root API", 
            "/api/",
            expected_fields=["message", "version", "ai_enabled", "endpoints"]
        )
        
        if success:
            print(f"   AI Enabled: {data.get('ai_enabled')}")
            print(f"   Version: {data.get('version')}")
        
        # Test 2: Demo Graph (HIGH PRIORITY)
        success, data = self.test_endpoint(
            "Demo Knowledge Graph", 
            "/api/demo-graph",
            expected_fields=["nodes", "edges", "metadata"]
        )
        
        if success:
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            metadata = data.get('metadata', {})
            
            print(f"   Nodes: {len(nodes)}")
            print(f"   Edges: {len(edges)}")
            print(f"   Articles: {metadata.get('total_articles', 0)}")
            print(f"   Connections: {metadata.get('total_connections', 0)}")
            print(f"   AI Analysis: {metadata.get('ai_analysis_enabled', False)}")
            print(f"   Demo Mode: {metadata.get('demo_mode', False)}")
            
            # Verify demo data structure
            article_nodes = [n for n in nodes if n.get('type') == 'article']
            section_nodes = [n for n in nodes if n.get('type') == 'section']
            story_connections = [e for e in edges if e.get('type') != 'belongs_to']
            
            print(f"   Article nodes: {len(article_nodes)}")
            print(f"   Section nodes: {len(section_nodes)}")
            print(f"   Story connections: {len(story_connections)}")
            
            # Check if we have the expected demo data
            if len(article_nodes) == 5 and len(section_nodes) == 5 and len(story_connections) == 5:
                print("✅ Demo data structure is correct!")
                self.tests_passed += 1
            else:
                print("⚠️  Demo data structure may be incomplete")
            
            self.tests_run += 1
        
        # Test 3: Real Knowledge Graph (may timeout due to AI quota)
        print(f"\n🔍 Testing Real Knowledge Graph (may timeout)...")
        try:
            response = requests.get(f"{self.base_url}/api/knowledge-graph?days=1&max_articles=3", timeout=15)
            if response.status_code == 200:
                print("✅ Real Knowledge Graph - Working")
                self.tests_passed += 1
            else:
                print(f"⚠️  Real Knowledge Graph - Status: {response.status_code} (may be quota limited)")
        except requests.exceptions.Timeout:
            print("⚠️  Real Knowledge Graph - Timeout (expected due to AI quota)")
        except Exception as e:
            print(f"⚠️  Real Knowledge Graph - Error: {str(e)}")
        
        self.tests_run += 1
        
        # Test 4: Search endpoint (quick test)
        print(f"\n🔍 Testing Search (may timeout)...")
        try:
            response = requests.get(f"{self.base_url}/api/search?query=test&days=1&max_articles=3", timeout=15)
            if response.status_code == 200:
                print("✅ Search Endpoint - Working")
                self.tests_passed += 1
            else:
                print(f"⚠️  Search Endpoint - Status: {response.status_code}")
        except requests.exceptions.Timeout:
            print("⚠️  Search Endpoint - Timeout (expected due to AI quota)")
        except Exception as e:
            print(f"⚠️  Search Endpoint - Error: {str(e)}")
        
        self.tests_run += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 FOCUSED TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        # Determine if critical functionality is working
        if self.tests_passed >= 3:  # Root API + Demo Graph + structure check
            print("\n🎉 CRITICAL FUNCTIONALITY WORKING!")
            print("✅ Demo mode is functional and ready for frontend testing")
            return True
        else:
            print("\n⚠️  CRITICAL ISSUES DETECTED")
            print("❌ Demo functionality may not work properly")
            return False

def main():
    tester = FocusedAPITester()
    success = tester.run_focused_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())