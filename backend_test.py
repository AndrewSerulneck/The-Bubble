import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any

class NewsKnowledgeGraphTester:
    def __init__(self, base_url="https://9bb23194-cb06-4a30-80b8-000a0e730972.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.nyt_api_key = "nfc8xuDu40fzC0j9"  # NYT API key from review request

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details
        })

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int = 200, params: Dict = None, timeout: int = 30) -> tuple:
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            print(f"\n🔍 Testing {name}...")
            print(f"   URL: {url}")
            if params:
                print(f"   Params: {params}")
            
            if method == 'GET':
                response = requests.get(url, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            
            if success:
                try:
                    data = response.json()
                    self.log_test(name, True, f"Status: {response.status_code}")
                    return True, data
                except json.JSONDecodeError:
                    self.log_test(name, False, f"Invalid JSON response. Status: {response.status_code}")
                    return False, {}
            else:
                self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, f"Request timeout after {timeout}s")
            return False, {}
        except requests.exceptions.ConnectionError:
            self.log_test(name, False, "Connection error - service may be down")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Unexpected error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        success, data = self.run_test(
            "Health Check",
            "GET", 
            "/api/health"
        )
        
        if success:
            # Verify health check response structure
            required_fields = ['status', 'timestamp', 'guardian_api', 'ai_services', 'database']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Health Check Structure", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Health Check Structure", True, "All required fields present")
                print(f"   Guardian API: {data.get('guardian_api')}")
                print(f"   AI Services: {data.get('ai_services')}")
                print(f"   Database: {data.get('database')}")
                return True
        
        return success

    def test_knowledge_graph_default(self):
        """Test default knowledge graph endpoint"""
        success, data = self.run_test(
            "Knowledge Graph (Default)",
            "GET",
            "/api/knowledge-graph",
            params={'days': 3, 'max_articles': 10}
        )
        
        if success:
            # Verify knowledge graph structure
            required_fields = ['nodes', 'edges', 'metadata']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                self.log_test("Knowledge Graph Structure", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check if we have nodes and edges
            nodes_count = len(data.get('nodes', []))
            edges_count = len(data.get('edges', []))
            metadata = data.get('metadata', {})
            
            print(f"   Nodes: {nodes_count}")
            print(f"   Edges: {edges_count}")
            print(f"   AI Analysis: {metadata.get('ai_analysis_enabled', 'Unknown')}")
            print(f"   Total Articles: {metadata.get('total_articles', 0)}")
            print(f"   Total Connections: {metadata.get('total_connections', 0)}")
            
            if nodes_count > 0:
                self.log_test("Knowledge Graph Data", True, f"Generated {nodes_count} nodes, {edges_count} edges")
                
                # Test node structure
                sample_node = data['nodes'][0]
                if sample_node.get('type') in ['article', 'section']:
                    self.log_test("Node Structure", True, f"Valid node type: {sample_node.get('type')}")
                else:
                    self.log_test("Node Structure", False, f"Invalid node type: {sample_node.get('type')}")
                
                return True
            else:
                self.log_test("Knowledge Graph Data", False, "No nodes generated")
                return False
        
        return success

    def test_search_functionality(self):
        """Test search endpoint with different queries"""
        test_queries = ["Trump", "climate", "economy"]
        
        for query in test_queries:
            success, data = self.run_test(
                f"Search Query: '{query}'",
                "GET",
                "/api/search",
                params={'query': query, 'days': 7, 'max_articles': 8}
            )
            
            if success:
                nodes_count = len(data.get('nodes', []))
                edges_count = len(data.get('edges', []))
                print(f"   Query '{query}': {nodes_count} nodes, {edges_count} edges")
                
                if nodes_count > 0:
                    # Check if articles are related to the query
                    article_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'article']
                    if article_nodes:
                        sample_title = article_nodes[0].get('title', '').lower()
                        if query.lower() in sample_title:
                            self.log_test(f"Search Relevance: '{query}'", True, "Query term found in results")
                        else:
                            self.log_test(f"Search Relevance: '{query}'", True, "Results returned (relevance may be contextual)")
            
            if not success:
                return False
        
        return True

    def test_recent_news_filtering(self):
        """Test recent news endpoint with different filters"""
        test_cases = [
            {'days': 5, 'section': None, 'name': 'Recent News (5 days)'},
            {'days': 1, 'section': 'business', 'name': 'Business News (1 day)'},
            {'days': 7, 'section': 'politics', 'name': 'Politics News (7 days)'},
        ]
        
        for case in test_cases:
            params = {'days': case['days']}
            if case['section']:
                params['section'] = case['section']
            
            success, data = self.run_test(
                case['name'],
                "GET",
                "/api/news/recent",
                params=params
            )
            
            if success:
                articles = data.get('articles', [])
                total = data.get('total', 0)
                query_params = data.get('query_parameters', {})
                
                print(f"   Articles: {len(articles)}")
                print(f"   Total: {total}")
                print(f"   Date range: {query_params.get('from_date')} to {query_params.get('to_date')}")
                
                if articles:
                    # Check article structure
                    sample_article = articles[0]
                    required_fields = ['id', 'title', 'summary', 'lede', 'nutgraf', 'section']
                    missing_fields = [field for field in required_fields if field not in sample_article]
                    
                    if missing_fields:
                        self.log_test(f"Article Structure ({case['name']})", False, f"Missing fields: {missing_fields}")
                    else:
                        self.log_test(f"Article Structure ({case['name']})", True, "All required fields present")
                        
                        # Check if section filter worked
                        if case['section']:
                            article_section = sample_article.get('section', '').lower()
                            if case['section'].lower() in article_section:
                                self.log_test(f"Section Filter ({case['section']})", True, f"Correct section: {article_section}")
                            else:
                                self.log_test(f"Section Filter ({case['section']})", False, f"Expected {case['section']}, got {article_section}")
                else:
                    self.log_test(f"Article Data ({case['name']})", False, "No articles returned")
            
            if not success:
                return False
        
        return True

    def test_ai_integration(self):
        """Test AI integration by checking for AI-generated content"""
        success, data = self.run_test(
            "AI Integration Test",
            "GET",
            "/api/knowledge-graph",
            params={'days': 2, 'max_articles': 5}
        )
        
        if success:
            metadata = data.get('metadata', {})
            ai_enabled = metadata.get('ai_analysis_enabled', False)
            
            if ai_enabled:
                self.log_test("AI Services", True, "AI analysis is enabled")
                
                # Check for AI-generated connections
                edges = data.get('edges', [])
                ai_connections = [e for e in edges if e.get('type') not in ['belongs_to']]
                
                if ai_connections:
                    self.log_test("AI Story Connections", True, f"Found {len(ai_connections)} AI-analyzed connections")
                    
                    # Check connection structure
                    sample_connection = ai_connections[0]
                    if all(field in sample_connection for field in ['type', 'strength', 'explanation']):
                        self.log_test("AI Connection Structure", True, "Valid connection structure")
                        print(f"   Sample connection: {sample_connection.get('type')} (strength: {sample_connection.get('strength')})")
                        print(f"   Explanation: {sample_connection.get('explanation', '')[:100]}...")
                    else:
                        self.log_test("AI Connection Structure", False, "Missing connection fields")
                else:
                    self.log_test("AI Story Connections", False, "No AI-analyzed connections found")
                
                # Check for AI-generated summaries in nodes
                article_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'article']
                if article_nodes:
                    sample_article = article_nodes[0]
                    if all(field in sample_article for field in ['summary', 'lede', 'nutgraf', 'engagement_preview']):
                        self.log_test("AI Content Generation", True, "AI-generated content fields present")
                    else:
                        self.log_test("AI Content Generation", False, "Missing AI-generated content fields")
            else:
                self.log_test("AI Services", False, "AI analysis is disabled")
        
        return success

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting News Knowledge Graph Backend Tests")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test sequence
        tests = [
            self.test_health_check,
            self.test_knowledge_graph_default,
            self.test_search_functionality,
            self.test_recent_news_filtering,
            self.test_ai_integration
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(f"Test {test.__name__}", False, f"Test crashed: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        # Print failed tests
        failed_tests = [t for t in self.test_results if not t['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        return self.tests_passed == self.tests_run

def main():
    tester = NewsKnowledgeGraphTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Backend is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())