#!/usr/bin/env python3
"""Test script to check visualization data loading."""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from examples.enhanced_3d_viewer import MemoryVisualization3D

def test_visualization_data():
    """Test if visualization is loading real data or sample data."""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating visualization instance...")
        viz = MemoryVisualization3D()
        
        logger.info("Setting up memory engine...")
        success = viz.setup_memory_engine('data/test_output')
        logger.info(f"Engine setup success: {success}")
        
        if viz.engine:
            logger.info("Engine found, checking data...")
            data = viz.load_memory_data()
            logger.info(f"Loaded nodes: {len(data['nodes'])}")
            logger.info(f"Loaded edges: {len(data['edges'])}")
            
            if data['nodes']:
                sample_node = data['nodes'][0]
                logger.info(f"Sample node content: {sample_node.get('content', 'No content')}")
                logger.info(f"Sample node type: {sample_node.get('memory_type', 'No type')}")
                logger.info(f"Sample node domain: {sample_node.get('domain', 'No domain')}")
                
                # Check if it's sample data by looking for characteristic sample content
                is_sample = any("Walking through the forest" in node.get('content', '') for node in data['nodes'])
                logger.info(f"Using sample data: {is_sample}")
            else:
                logger.info("No nodes found")
        else:
            logger.info("No engine available")
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_data()
    sys.exit(0 if success else 1)
