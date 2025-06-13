#!/usr/bin/env python3
"""
Test GPU-Accelerated Memory Visualization with Real merX Data

Integration test to verify GPU acceleration performance with actual
merX memory database containing 50k+ records.
"""

import sys
import os
from pathlib import Path
import time
import logging
import numpy as np
from datetime import datetime

# Add merX to path
sys.path.append(str(Path(__file__).parent.parent))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
from examples.gpu_accelerated_viewer import GPUAcceleratedMemoryViewer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_merx_data(mex_file_path: str = "data/test_output/hp_mini.mex"):
    """Load real merX data from test database"""
    logger.info(f"🔄 Loading merX data from {mex_file_path}")
    
    # Initialize merX engine
    engine = EnhancedMemoryEngineFactory.create_engine(
        data_path=mex_file_path,
        ram_capacity=50000
    )
    
    # Load the test database and get data from RAMX
    if os.path.exists(mex_file_path):
        logger.info(f"✅ Loading merX database from {mex_file_path}")
        
        # Get RAMX instance directly
        ramx = getattr(engine.storage, "ramx", None)
        if ramx:
            ramx_nodes = ramx.get_all_nodes()
            logger.info(f"✅ Loaded {len(ramx_nodes)} nodes from RAMX")
        else:
            logger.error("❌ Could not access RAMX from engine")
            return None
    else:
        logger.error(f"❌ merX file not found: {mex_file_path}")
        return None
    
    # Convert to visualization format
    nodes = []
    edges = []
    
    for i, ramx_node in enumerate(ramx_nodes):
        node = {
            'id': f"memory_{i}",
            'content': ramx_node.content[:200],  # Limit content for performance
            'domain': ramx_node.tags[0] if ramx_node.tags else 'unknown',
            'memory_type': ramx_node.node_type,
            'age_hours': (time.time() - ramx_node.timestamp) / 3600,
            'decay_factor': ramx_node.activation,
            'access_count': 1,  # Default value
            'timestamp': str(datetime.fromtimestamp(ramx_node.timestamp)),
            'visual': {
                'size': max(3, min(15, 3 + ramx_node.activation * 5)),
                'color': f"rgba({hash(ramx_node.node_type) % 255},{hash(ramx_node.tags[0] if ramx_node.tags else 'default') % 255},150,0.8)"
            }
        }
        nodes.append(node)
        
        # Create edges based on RAMX links
        for target_id, (weight, link_type) in ramx_node.links.items():
            # Find the target node index
            target_idx = None
            for j, other_node in enumerate(ramx_nodes):
                if other_node.id == target_id:
                    target_idx = j
                    break
            
            if target_idx is not None:
                edges.append({
                    'source': f"memory_{i}",
                    'target': f"memory_{target_idx}",
                    'weight': weight
                })
                
                # Limit edges for performance
                if len(edges) >= 10000:
                    break
        
        if len(edges) >= 10000:
            break
    
    logger.info(f"📊 Prepared {len(nodes)} nodes and {len(edges)} edges for visualization")
    return {'nodes': nodes, 'edges': edges}

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    
    logger.info("🚀 Starting GPU-Accelerated Memory Visualization Performance Benchmark")
    logger.info("=" * 80)
    
    # Load real merX data
    start_time = time.time()
    data = load_real_merx_data()
    
    if data is None:
        logger.error("❌ Failed to load merX data - creating synthetic data for testing")
        # Create large synthetic dataset for testing
        import numpy as np
        np.random.seed(42)
        
        domains = ['artificial_intelligence', 'computer_science', 'biology', 'physics', 'chemistry', 
                  'mathematics', 'literature', 'history', 'psychology', 'philosophy']
        memory_types = ['working', 'episodic', 'semantic', 'procedural']
        
        nodes = []
        for i in range(50000):  # 50k nodes for stress test
            node = {
                'id': f"synthetic_{i}",
                'content': f"Synthetic memory content {i} with detailed information about various topics",
                'domain': np.random.choice(domains),
                'memory_type': np.random.choice(memory_types),
                'age_hours': np.random.exponential(24),
                'decay_factor': np.random.beta(2, 5),
                'access_count': np.random.poisson(3),
                'visual': {
                    'size': np.random.uniform(3, 12),
                }
            }
            nodes.append(node)
        
        # Create edges (limited for performance)
        edges = []
        for i in range(0, 50000, 50):  # Every 50th node
            for j in range(i+1, min(i+10, 50000)):
                edges.append({
                    'source': f"synthetic_{i}",
                    'target': f"synthetic_{j}",
                    'weight': np.random.uniform(0.1, 1.0)
                })
                
                if len(edges) >= 15000:  # Limit total edges
                    break
            if len(edges) >= 15000:
                break
        
        data = {'nodes': nodes, 'edges': edges}
        logger.info(f"🧪 Created synthetic dataset: {len(nodes)} nodes, {len(edges)} edges")
    
    load_time = time.time() - start_time
    logger.info(f"⏱️  Data loading completed in {load_time:.3f}s")
    
    # Initialize GPU viewer
    viewer = GPUAcceleratedMemoryViewer()
    
    # Performance Test 1: Semantic PCA Layout with LOD
    logger.info("\n🔬 Test 1: Semantic PCA + LOD (Recommended)")
    test1_start = time.time()
    
    viz_file1 = viewer.create_gpu_accelerated_visualization(
        data,
        layout_type="semantic_pca",
        enable_lod=True,
        camera_distance=50.0
    )
    
    test1_time = time.time() - test1_start
    logger.info(f"✅ Test 1 completed in {test1_time:.3f}s - File: {viz_file1}")
    
    # Performance Test 2: Sphere Layout with Aggressive LOD  
    logger.info("\n🔬 Test 2: Sphere Layout + Aggressive LOD")
    test2_start = time.time()
    
    viz_file2 = viewer.create_gpu_accelerated_visualization(
        data,
        layout_type="sphere",
        enable_lod=True,
        camera_distance=100.0  # Higher distance = more aggressive LOD
    )
    
    test2_time = time.time() - test2_start
    logger.info(f"✅ Test 2 completed in {test2_time:.3f}s - File: {viz_file2}")
    
    # Performance Test 3: Spiral Layout with Moderate LOD
    logger.info("\n🔬 Test 3: Spiral Layout + Moderate LOD")
    test3_start = time.time()
    
    viz_file3 = viewer.create_gpu_accelerated_visualization(
        data,
        layout_type="spiral",
        enable_lod=True,
        camera_distance=75.0
    )
    
    test3_time = time.time() - test3_start
    logger.info(f"✅ Test 3 completed in {test3_time:.3f}s - File: {viz_file3}")
    
    # Performance Test 4: LOD Comparison Visualization
    logger.info("\n🔬 Test 4: LOD Performance Comparison")
    test4_start = time.time()
    
    comparison_file = viewer.create_performance_comparison(data)
    
    test4_time = time.time() - test4_start
    logger.info(f"✅ Test 4 completed in {test4_time:.3f}s - File: {comparison_file}")
    
    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("📊 PERFORMANCE BENCHMARK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"📈 Dataset Size: {len(data['nodes']):,} nodes, {len(data['edges']):,} edges")
    logger.info(f"⏱️  Total Runtime: {total_time:.3f}s")
    logger.info(f"🔄 Data Loading: {load_time:.3f}s")
    logger.info(f"🎯 Test 1 (Semantic PCA): {test1_time:.3f}s")
    logger.info(f"🌐 Test 2 (Sphere): {test2_time:.3f}s") 
    logger.info(f"🌀 Test 3 (Spiral): {test3_time:.3f}s")
    logger.info(f"📊 Test 4 (Comparison): {test4_time:.3f}s")
    
    # Performance analysis
    avg_render_time = (test1_time + test2_time + test3_time) / 3
    nodes_per_second = len(data['nodes']) / avg_render_time
    
    logger.info(f"⚡ Average Render Time: {avg_render_time:.3f}s")
    logger.info(f"🚀 Rendering Speed: {nodes_per_second:,.0f} nodes/second")
    
    if avg_render_time < 3.0:
        logger.info("✅ EXCELLENT: Real-time rendering achieved!")
    elif avg_render_time < 6.0:
        logger.info("⚡ GOOD: Smooth interaction performance")
    elif avg_render_time < 12.0:
        logger.info("⚠️  MODERATE: Acceptable for batch visualization")
    else:
        logger.info("🐌 SLOW: Consider optimization or smaller datasets")
    
    # File size analysis
    from pathlib import Path
    for i, file_path in enumerate([viz_file1, viz_file2, viz_file3, comparison_file], 1):
        if os.path.exists(file_path):
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            logger.info(f"📁 Test {i} File Size: {size_mb:.2f} MB")
    
    logger.info("=" * 80)
    logger.info("🎉 GPU-Accelerated Visualization Benchmark Complete!")
    
    return {
        'total_time': total_time,
        'load_time': load_time,
        'render_times': [test1_time, test2_time, test3_time, test4_time],
        'nodes_per_second': nodes_per_second,
        'files': [viz_file1, viz_file2, viz_file3, comparison_file]
    }

def test_memory_usage():
    """Test memory usage with different configurations"""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available, skipping memory usage test")
        return
        
    logger.info("\n🧠 Memory Usage Analysis")
    logger.info("-" * 40)
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"📊 Initial Memory Usage: {initial_memory:.2f} MB")
    
    # Load moderate dataset
    np.random.seed(42)
    
    nodes = []
    for i in range(10000):  # 10k for memory test
        node = {
            'id': f"mem_test_{i}",
            'content': f"Memory test content {i}" * 10,  # Larger content
            'domain': np.random.choice(['ai', 'cs', 'bio']),
            'memory_type': 'episodic',
            'age_hours': i * 0.1,
            'decay_factor': 0.5,
            'access_count': i % 100,
        }
        nodes.append(node)
    
    data = {'nodes': nodes, 'edges': []}
    
    viewer = GPUAcceleratedMemoryViewer()
    
    # Test different LOD levels
    for lod_distance in [25, 50, 100, 200]:
        import gc
        gc.collect()  # Clean up before test
        
        memory_before = process.memory_info().rss / 1024 / 1024
        
        viewer.create_gpu_accelerated_visualization(
            data,
            layout_type="sphere",
            enable_lod=True,
            camera_distance=lod_distance
        )
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        logger.info(f"🔍 LOD Distance {lod_distance}: {memory_used:.2f} MB (+{memory_used:.1f})")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    total_used = final_memory - initial_memory
    
    logger.info(f"💾 Total Memory Used: {total_used:.2f} MB")
    logger.info(f"📈 Final Memory Usage: {final_memory:.2f} MB")

if __name__ == "__main__":
    try:
        # Run main performance benchmark
        results = run_performance_benchmark()
        
        # Run memory usage test
        test_memory_usage()
        
        logger.info("\n🎯 Recommendations for 50k+ Node Performance:")
        logger.info("   1. Use 'semantic_pca' layout for best semantic clustering")
        logger.info("   2. Enable LOD with camera_distance >= 50 for smooth interaction")
        logger.info("   3. Limit edges to < 10k for optimal performance")
        logger.info("   4. Use quality levels 0.6-0.8 for best visual/performance balance")
        logger.info("   5. Monitor FPS with built-in performance counter")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
