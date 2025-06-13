#!/usr/bin/env python3
"""
Test script to verify domain distribution fix
"""

from examples.database_viewer import EnhancedDatabaseViewer
import os

def test_domain_distribution():
    print('Testing domain distribution fix...')
    viewer = EnhancedDatabaseViewer()
    data = viewer.load_memory_data()

    print(f'Total nodes: {data["stats"]["total_nodes"]}')
    print(f'Total edges: {data["stats"]["total_edges"]}')
    print('Domain distribution:')
    for domain, count in data['stats']['domains'].items():
        print(f'  {domain}: {count}')
    print()

    # Check first few nodes to see their tags
    print('Sample node tags (first 10 nodes):')
    for i, node in enumerate(data['nodes'][:10]):
        print(f'  Node {i}: {node.get("tags", [])}')
    
    print()
    
    # Check if the domains are now being recognized (not "general")
    known_domains = {
        "artificial_intelligence", "computer_science", "biology", "physics", 
        "chemistry", "mathematics", "literature", "history", "psychology", "philosophy"
    }
    
    recognized_domains = set(data['stats']['domains'].keys())
    missing_domains = known_domains - recognized_domains
    
    print(f"Recognized domains: {len(recognized_domains)}")
    print(f"Expected domains: {len(known_domains)}")
    
    if missing_domains:
        print(f"Missing domains: {missing_domains}")
    else:
        print("✅ All expected domains are being recognized!")

if __name__ == "__main__":
    test_domain_distribution()
