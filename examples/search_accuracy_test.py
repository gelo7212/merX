#!/usr/bin/env python3
"""
Search Accuracy Test with Tree-of-Trees Structure

This script creates a comprehensive memory structure with multiple knowledge domains
arranged in a tree-of-trees pattern and tests search accuracy across different
query types and recall methods.
"""

import os
import sys
import json
import time
import random
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from uuid import uuid4

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
from src.container.enhanced_di_container import create_enhanced_container

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
NUM_DOMAINS = 5  # Number of knowledge domains (trees)
NODES_PER_DOMAIN = 100  # Nodes per domain tree
QUERIES_PER_DOMAIN = 5  # Test queries per domain
CROSS_REFERENCES = 25  # Links between different domain trees

# Knowledge domains for testing
DOMAINS = {
    "artificial_intelligence": {
        "topics": ["machine learning", "neural networks", "deep learning", "computer vision", "nlp"],
        "sample_content": [
            "Machine learning algorithms enable computers to learn patterns from data without explicit programming.",
            "Neural networks consist of interconnected nodes that process information in layers.",
            "Deep learning uses multi-layer neural networks to solve complex pattern recognition tasks.",
            "Computer vision systems analyze and interpret visual information from images and videos.",
            "Natural language processing enables computers to understand and generate human language."
        ],
        "test_queries": [
            "machine learning algorithms",
            "neural network training", 
            "deep learning models",
            "computer vision systems",
            "artificial intelligence applications"
        ]
    },
    "computer_science": {
        "topics": ["algorithms", "data structures", "programming", "software engineering", "databases"],
        "sample_content": [
            "Algorithms are step-by-step procedures for solving computational problems efficiently.",
            "Data structures organize and store data to enable efficient access and modification.",
            "Programming languages provide syntax and semantics for creating software applications.",
            "Software engineering applies systematic approaches to develop large-scale software systems.",
            "Database systems store, organize, and retrieve large amounts of structured information."
        ],
        "test_queries": [
            "programming languages",
            "data structure performance", 
            "software development",
            "algorithm complexity",
            "code optimization"
        ]
    },
    "mathematics": {
        "topics": ["calculus", "linear algebra", "statistics", "geometry", "number theory"],
        "sample_content": [
            "Calculus studies continuous change through derivatives and integrals.",
            "Linear algebra deals with vector spaces and linear transformations between them.",
            "Statistics analyzes data to understand patterns and make informed decisions.",
            "Geometry explores properties and relationships of points, lines, shapes, and spaces.",
            "Number theory investigates properties and relationships of integers and rational numbers."
        ],
        "test_queries": [
            "calculus derivatives",
            "linear algebra matrices",
            "statistical analysis", 
            "geometric proofs",
            "number theory applications"
        ]
    },
    "physics": {
        "topics": ["mechanics", "thermodynamics", "electromagnetism", "relativity", "quantum"],
        "sample_content": [
            "Classical mechanics describes the motion of objects from projectiles to spacecraft.",
            "Thermodynamics studies heat, work, and energy transfer in physical systems.",
            "Electromagnetism explains electric and magnetic phenomena and their interactions.",
            "Relativity theory describes space, time, and gravity at cosmic scales.",
            "Quantum mechanics governs the behavior of matter and energy at atomic scales."
        ],
        "test_queries": [
            "energy conservation",
            "electromagnetic forces",
            "theory of relativity",
            "quantum mechanics",
            "thermodynamic processes"
        ]
    },
    "biology": {
        "topics": ["genetics", "evolution", "ecology", "molecular biology", "physiology"],
        "sample_content": [
            "Genetics studies heredity and variation in living organisms through DNA analysis.",
            "Evolution explains how species change over time through natural selection.",
            "Ecology examines interactions between organisms and their environments.",
            "Molecular biology investigates biological processes at the molecular level.",
            "Physiology studies the functions and mechanisms of living systems."
        ],
        "test_queries": [
            "DNA replication process",
            "genetic inheritance patterns",
            "evolutionary biology",
            "ecological systems",
            "cellular physiology"
        ]
    }
}

class SearchAccuracyTester:
    """Tests search accuracy across different recall methods and query types."""
    
    def __init__(self):
        self.results = {
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "pass_rate": 0.0,
                "failed_tests": 0
            },
            "accuracy_metrics": {
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "std_precision": 0.0,
                "std_recall": 0.0,
                "std_f1": 0.0
            },
            "performance_metrics": {
                "avg_response_time_ms": 0.0,
                "std_response_time_ms": 0.0,
                "min_response_time_ms": float('inf'),
                "max_response_time_ms": 0.0
            },
            "tree_structure": {
                "domains": {},
                "total_nodes": 0,
                "tree_count": 0,
                "cross_references": 0
            },
            "individual_results": []
        }
        self.engine = None
        self.domain_nodes = {}  # domain -> list of node IDs
        
    def setup_test_environment(self):
        """Set up the memory engine and test environment."""
        logger.info("Setting up test environment...")
        
        # Create enhanced memory engine        container = create_enhanced_container()
        factory = EnhancedMemoryEngineFactory(container)
        
        # Use test-specific data directory
        test_dir = "data/test_data"
        os.makedirs(test_dir, exist_ok=True)
        
        self.engine = factory.create_memory_engine(
            data_file=f"{test_dir}/memory",
            enable_versioning=True,
            enable_decay=False,  # Disable for consistent testing
            enable_distributed_storage=False
        )
        
        logger.info("Memory engine initialized successfully")
        
    def create_tree_of_trees_structure(self):
        """Create a tree-of-trees memory structure with multiple knowledge domains."""
        logger.info("Creating tree-of-trees memory structure...")
        
        all_node_ids = []
        
        # Create each domain tree
        for domain_name, domain_data in DOMAINS.items():
            logger.info(f"Creating {domain_name} tree...")
            domain_node_ids = []
            
            # Create nodes for this domain
            for i in range(NODES_PER_DOMAIN):
                topic = random.choice(domain_data["topics"])
                content_template = random.choice(domain_data["sample_content"])
                
                # Generate varied content
                content = f"{content_template} This relates to {topic} in {domain_name} domain. Node {i+1} provides detailed information about this concept."
                
                # Create tags
                tags = [domain_name, topic, "test_data", f"node_{i+1}"]
                if i < 20:  # First 20 nodes are "important"
                    tags.append("important")
                
                # Select related nodes within domain (tree structure)
                related_ids = []
                if domain_node_ids:  # Connect to existing nodes in this domain
                    num_connections = min(random.randint(1, 3), len(domain_node_ids))
                    related_ids = random.sample(domain_node_ids, num_connections)
                
                # Insert memory node
                node_id = self.engine.insert_memory(
                    content=content,
                    node_type="concept",
                    tags=tags,
                    related_ids=related_ids
                )
                
                domain_node_ids.append(node_id)
                all_node_ids.append(node_id)
            
            self.domain_nodes[domain_name] = domain_node_ids
            self.results["tree_structure"]["domains"][domain_name] = {
                "node_count": len(domain_node_ids),
                "tree_depth": min(10, NODES_PER_DOMAIN // 10),  # Estimated depth
                "topics": domain_data["topics"]
            }
            
            logger.info(f"Created {len(domain_node_ids)} nodes for {domain_name}")
        
        # Create cross-references between domain trees
        logger.info("Creating cross-references between domain trees...")
        cross_ref_count = 0
        
        for _ in range(CROSS_REFERENCES):
            # Select two different domains
            domain_names = list(self.domain_nodes.keys())
            if len(domain_names) >= 2:
                domain1, domain2 = random.sample(domain_names, 2)
                
                # Select random nodes from each domain
                node1 = random.choice(self.domain_nodes[domain1])
                node2 = random.choice(self.domain_nodes[domain2])
                
                # Create bidirectional link
                try:
                    self.engine.storage.memory_linker.add_link(node1, node2, "cross_domain")
                    cross_ref_count += 1
                except:
                    pass  # Link might already exist
        
        self.results["tree_structure"]["total_nodes"] = len(all_node_ids)
        self.results["tree_structure"]["tree_count"] = len(DOMAINS)
        self.results["tree_structure"]["cross_references"] = cross_ref_count
        
        logger.info(f"Created tree-of-trees structure with {len(all_node_ids)} total nodes across {len(DOMAINS)} domains")
        logger.info(f"Added {cross_ref_count} cross-domain references")
        
    def calculate_metrics(self, retrieved_nodes: List[Any], expected_domain: str, query: str) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for a query result."""
        if not retrieved_nodes:
            return 0.0, 0.0, 0.0
        
        # Count relevant results (nodes from the expected domain)
        relevant_results = 0
        for node in retrieved_nodes:
            if hasattr(node, 'tags') and expected_domain in node.tags:
                relevant_results += 1
            elif hasattr(node, 'data') and hasattr(node.data, 'tags') and expected_domain in node.data.tags:
                relevant_results += 1
        
        # Calculate metrics
        precision = relevant_results / len(retrieved_nodes) if retrieved_nodes else 0.0
        
        # For recall, estimate total relevant nodes in domain
        total_relevant = len(self.domain_nodes.get(expected_domain, []))
        recall = relevant_results / total_relevant if total_relevant > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def run_search_tests(self):
        """Run comprehensive search accuracy tests."""
        logger.info("Running search accuracy tests...")
        
        all_results = []
        
        for domain_name, domain_data in DOMAINS.items():
            logger.info(f"Testing {domain_name} domain...")
            
            for query in domain_data["test_queries"]:
                logger.info(f"Testing query: '{query}'")
                
                # Test content-based recall
                start_time = time.time()
                try:
                    results = self.engine.recall_memories(query, max_results=50)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Calculate metrics
                    precision, recall, f1 = self.calculate_metrics(results, domain_name, query)
                    
                    # Record result
                    result = {
                        "query": query,
                        "domain": domain_name,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "response_time_ms": response_time,
                        "total_results": len(results),
                        "relevant_results": int(precision * len(results)) if results else 0
                    }
                    
                    all_results.append(result)
                    
                    # Update performance metrics
                    self.results["performance_metrics"]["min_response_time_ms"] = min(
                        self.results["performance_metrics"]["min_response_time_ms"], 
                        response_time
                    )
                    self.results["performance_metrics"]["max_response_time_ms"] = max(
                        self.results["performance_metrics"]["max_response_time_ms"], 
                        response_time
                    )
                    
                    logger.info(f"Query '{query}': P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}, Time={response_time:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Error testing query '{query}': {e}")
                    # Record failed test
                    result = {
                        "query": query,
                        "domain": domain_name,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "response_time_ms": 0.0,
                        "total_results": 0,
                        "relevant_results": 0
                    }
                    all_results.append(result)
        
        self.results["individual_results"] = all_results
        
        # Calculate summary statistics
        if all_results:
            precisions = [r["precision"] for r in all_results]
            recalls = [r["recall"] for r in all_results]
            f1s = [r["f1"] for r in all_results]
            times = [r["response_time_ms"] for r in all_results]
            
            self.results["accuracy_metrics"]["avg_precision"] = sum(precisions) / len(precisions)
            self.results["accuracy_metrics"]["avg_recall"] = sum(recalls) / len(recalls)
            self.results["accuracy_metrics"]["avg_f1"] = sum(f1s) / len(f1s)
            
            # Calculate standard deviations
            import statistics
            if len(precisions) > 1:
                self.results["accuracy_metrics"]["std_precision"] = statistics.stdev(precisions)
                self.results["accuracy_metrics"]["std_recall"] = statistics.stdev(recalls)
                self.results["accuracy_metrics"]["std_f1"] = statistics.stdev(f1s)
            
            self.results["performance_metrics"]["avg_response_time_ms"] = sum(times) / len(times)
            if len(times) > 1:
                self.results["performance_metrics"]["std_response_time_ms"] = statistics.stdev(times)
            
            # Count passed tests (F1 > 0.5 is considered "passed")
            passed_tests = sum(1 for r in all_results if r["f1"] > 0.5)
            self.results["test_summary"]["total_tests"] = len(all_results)
            self.results["test_summary"]["passed_tests"] = passed_tests
            self.results["test_summary"]["failed_tests"] = len(all_results) - passed_tests
            self.results["test_summary"]["pass_rate"] = passed_tests / len(all_results) if all_results else 0.0
        
        logger.info(f"Search accuracy tests completed. Pass rate: {self.results['test_summary']['pass_rate']:.2%}")
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = "data/search_accuracy_test/search_accuracy_results.json"
        
        # Add timestamp
        self.results["timestamp"] = datetime.now().isoformat()
        self.results["test_configuration"] = {
            "num_domains": NUM_DOMAINS,
            "nodes_per_domain": NODES_PER_DOMAIN,
            "queries_per_domain": QUERIES_PER_DOMAIN,
            "cross_references": CROSS_REFERENCES
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "="*80)
        print("SEARCH ACCURACY TEST RESULTS")
        print("="*80)
        
        # Test summary
        summary = self.results["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed Tests: {summary['passed_tests']}")
        print(f"Failed Tests: {summary['failed_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        
        # Accuracy metrics
        metrics = self.results["accuracy_metrics"]
        print(f"\nAccuracy Metrics:")
        print(f"  Average Precision: {metrics['avg_precision']:.3f} ± {metrics['std_precision']:.3f}")
        print(f"  Average Recall: {metrics['avg_recall']:.3f} ± {metrics['std_recall']:.3f}")
        print(f"  Average F1 Score: {metrics['avg_f1']:.3f} ± {metrics['std_f1']:.3f}")
        
        # Performance metrics
        perf = self.results["performance_metrics"]
        print(f"\nPerformance Metrics:")
        print(f"  Average Response Time: {perf['avg_response_time_ms']:.2f} ± {perf['std_response_time_ms']:.2f} ms")
        print(f"  Min Response Time: {perf['min_response_time_ms']:.2f} ms")
        print(f"  Max Response Time: {perf['max_response_time_ms']:.2f} ms")
        
        # Tree structure
        structure = self.results["tree_structure"]
        print(f"\nTree Structure:")
        print(f"  Total Nodes: {structure['total_nodes']}")
        print(f"  Domain Trees: {structure['tree_count']}")
        print(f"  Cross References: {structure['cross_references']}")
        
        print("="*80)

def main():
    """Main function to run the search accuracy test."""
    logger.info("Starting Search Accuracy Test with Tree-of-Trees Structure")
    
    tester = SearchAccuracyTester()
    
    try:
        # Set up test environment
        tester.setup_test_environment()
        
        # Create tree-of-trees memory structure
        tester.create_tree_of_trees_structure()
        
        # Run search tests
        tester.run_search_tests()
        
        # Save and display results
        tester.save_results()
        tester.print_summary()
        
        logger.info("Search accuracy test completed successfully!")
        
    except Exception as e:
        logger.error(f"Search accuracy test failed: {e}")
        raise

if __name__ == "__main__":
    main()
