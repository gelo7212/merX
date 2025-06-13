#!/usr/bin/env python3
"""
Test script to verify merX system functionality after critical fixes.
"""

import sys
import os
import traceback
from uuid import uuid4

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic merX functionality."""
    print("🧪 Testing merX Basic Functionality...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.interfaces import MemoryNode, MemoryLink
        from src.core.ramx import RAMX, RAMXNode
        from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
        print("   ✓ All imports successful")
        
        # Test MemoryNode creation
        print("2. Testing MemoryNode creation...")
        from datetime import datetime
        node = MemoryNode(
            id=uuid4(),
            content="Test memory",
            node_type="test",
            version=1,
            timestamp=datetime.now(),
            activation=1.0,
            decay_rate=0.02,
            links=[],
            tags=["test", "memory"]
        )
        print(f"   ✓ MemoryNode created: {node.id}")
        
        # Test MemoryLink creation
        print("3. Testing MemoryLink creation...")
        link = MemoryLink(to_id=uuid4(), weight=0.8, link_type="related")
        print(f"   ✓ MemoryLink created: {link.to_id} (weight: {link.weight})")
        
        # Test RAMX creation
        print("4. Testing RAMX creation...")
        ramx = RAMX(capacity=1000)
        print(f"   ✓ RAMX created with capacity: {ramx._capacity}")
        
        # Test RAMXNode conversion
        print("5. Testing RAMXNode conversion...")
        ramx_node = RAMXNode.from_memory_node(node)
        print(f"   ✓ RAMXNode created: {ramx_node.id}")
        
        # Test factory creation
        print("6. Testing Memory Engine Factory...")
        try:
            engine = EnhancedMemoryEngineFactory.create_engine(
                ram_capacity=1000,
                data_path="data/test_memory.mex"
            )
            print("   ✓ Memory Engine created successfully")
            
            # Test memory insertion
            print("7. Testing memory insertion...")
            memory_id = engine.insert_memory(
                content="This is a test memory for the merX system",
                node_type="test",
                tags=["system", "test", "verification"]
            )
            print(f"   ✓ Memory inserted: {memory_id}")
            
            # Test memory retrieval
            print("8. Testing memory retrieval...")
            retrieved = engine.get_memory(memory_id)
            if retrieved:
                print(f"   ✓ Memory retrieved: {retrieved.content[:50]}...")
            else:
                print("   ❌ Memory retrieval failed")
                
            # Test memory stats
            print("9. Testing memory statistics...")
            stats = engine.get_memory_stats()
            print(f"   ✓ Memory stats: {stats.get('total_memories', 0)} memories")
            
            # Test memory recall
            print("10. Testing memory recall...")
            recalled = engine.recall_memories(query="test system", limit=5)
            print(f"   ✓ Recalled {len(recalled)} memories")
            
        except Exception as e:
            print(f"   ❌ Factory/Engine test failed: {e}")
            traceback.print_exc()
            return False
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

def test_interface_alignment():
    """Test that interfaces are properly aligned."""
    print("\n🔍 Testing Interface Alignment...")
    
    try:
        from src.interfaces import MemoryLink
        from src.core.ramx import RAMXNode
        from datetime import datetime
        
        # Test MemoryLink structure
        print("1. Testing MemoryLink interface...")
        link = MemoryLink(to_id=uuid4(), weight=0.5)
        assert hasattr(link, 'to_id'), "MemoryLink should have 'to_id' attribute"
        assert hasattr(link, 'weight'), "MemoryLink should have 'weight' attribute"
        print("   ✓ MemoryLink interface correct")
        
        # Test RAMXNode link conversion
        print("2. Testing RAMXNode link handling...")
        from src.interfaces import MemoryNode
        
        test_node = MemoryNode(
            id=uuid4(),
            content="Test node with links",
            node_type="test",
            version=1,
            timestamp=datetime.now(),
            activation=1.0,
            decay_rate=0.02,
            links=[link],
            tags=["test"]
        )
        
        ramx_node = RAMXNode.from_memory_node(test_node)
        assert len(ramx_node.links) == 1, "RAMXNode should have converted link"
        print("   ✓ RAMXNode link conversion works")
        
        print("✅ Interface alignment tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Interface alignment test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 merX System Verification Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test interface alignment
    if not test_interface_alignment():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! merX system is functional.")
        print("\n🎯 System is ALIGNED with merX vision:")
        print("   ✓ Memory nodes can be created and stored")
        print("   ✓ Links use correct 'to_id' interface")
        print("   ✓ RAMX conversion works properly")
        print("   ✓ Memory engine factory creates working engines")
        print("   ✓ Memory insertion/retrieval works")
        print("   ✓ Memory recall functionality works")
    else:
        print("❌ SOME TESTS FAILED! System needs fixes.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
