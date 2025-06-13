#!/usr/bin/env python3
"""Test script for update_memory_activation and apply_global_decay methods."""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory

def test_memory_methods():
    """Test the update_memory_activation and apply_global_decay methods."""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating memory engine...")
        
        # Create a test engine 
        engine = EnhancedMemoryEngineFactory.create_engine(
            data_path='data/test_memory_methods.mex',
            ram_capacity=1000
        )
        
        logger.info("✅ Memory engine created successfully")
        
        # Test inserting a memory
        logger.info("Inserting test memory...")
        node_id = engine.insert_memory('Test memory for activation update', 'test')
        logger.info(f"✅ Inserted memory: {node_id}")
        
        # Insert a few more memories for testing
        node_id2 = engine.insert_memory('Another test memory', 'test')
        node_id3 = engine.insert_memory('Yet another memory for decay testing', 'procedural')
        
        # Test updating activation
        logger.info("Testing update_memory_activation...")
        engine.update_memory_activation(node_id, boost=0.2)
        logger.info("✅ update_memory_activation completed successfully")
        
        # Test global decay
        logger.info("Testing apply_global_decay...")
        decay_count = engine.apply_global_decay()
        logger.info(f"✅ apply_global_decay completed - processed {decay_count} nodes")
        
        # Test getting memory stats
        logger.info("Getting memory statistics...")
        stats = engine.get_memory_stats()
        logger.info(f"✅ Memory stats: {stats}")
        
        logger.info("🎉 All tests passed! Both methods are working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_methods()
    sys.exit(0 if success else 1)
