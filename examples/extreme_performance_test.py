#!/usr/bin/env python3
"""
merX Extreme Performance Test - Stress tests the memory system at scale

This script tests:
1. Creating an enhanced memory engine with maximized capacity
2. Inserting 100,000+ memory nodes with various relationships
3. Testing recall performance under extreme load
4. Measuring memory consumption and CPU usage
5. Testing concurrent operations
6. Evaluating distributed storage capabilities

This provides a comprehensive stress test of the merX system's performance and scalability.
"""

import os
import sys
import time
import logging
import random
import gc
import multiprocessing
import threading
import traceback
import psutil
import json
import queue
from typing import List, Dict, Any, Tuple, Set, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
from src.container.enhanced_di_container import create_enhanced_container
from src.engine.recall_engine import RecallEngine
from src.core.ramx import RAMX, RAMXNode
from src.utils.helpers import MemoryProfiler, benchmark_operation
from src.interfaces import MemoryNode, MemoryLink
from src.core.memory_io_orchestrator import MemoryIOOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="test_output_results.log",
)
logger = logging.getLogger(__name__)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# Constants
NUM_NODES = 50000  # Total nodes to create
BATCH_SIZE = 1000  # Batch size for insertion
NUM_DOMAINS = 10  # Number of knowledge domains
TOPICS_PER_DOMAIN = 20  # Topics per domain
FACTS_PER_TOPIC = 100  # Facts per topic
MAX_LINKS_PER_NODE = 5  # Maximum links per node
MAX_TAGS_PER_NODE = 6  # Maximum tags per node
RECALL_QUERIES = 100  # Number of recall operations to test
CONCURRENT_OPERATIONS = 8  # Number of concurrent operations
WRITE_BUFFER_SIZE = 100000  # Number of nodes to buffer before writing to disk

# Test data directory
TEST_DIR = "data/test_output"
TEST_FILE = f"{TEST_DIR}/hp_mini.mex"

# Global profiler
profiler = MemoryProfiler()


class PerformanceMetrics:
    """Tracks and reports performance metrics."""

    def __init__(self):
        self.metrics = {
            "insert": {"total_time": 0, "operations": 0, "nodes_per_second": 0},
            "retrieval": {"total_time": 0, "operations": 0, "nodes_per_second": 0},
            "content_recall": {
                "total_time": 0,
                "operations": 0,
                "avg_time_ms": 0,
                "avg_results": 0,
            },
            "tag_recall": {
                "total_time": 0,
                "operations": 0,
                "avg_time_ms": 0,
                "avg_results": 0,
            },
            "spreading_activation": {
                "total_time": 0,
                "operations": 0,
                "avg_time_ms": 0,
                "avg_results": 0,
            },
            "memory_usage": {"peak_mb": 0, "final_mb": 0},
            "storage": {"file_size_mb": 0, "index_size_mb": 0, "bytes_per_node": 0},
            "concurrent": {"sequential_time": 0, "parallel_time": 0, "speedup": 0},
        }
        self.process = psutil.Process(os.getpid())

    def update_memory_usage(self):
        """Update memory usage metrics."""
        # Force garbage collection to get accurate readings
        gc.collect()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        if memory_mb > self.metrics["memory_usage"]["peak_mb"]:
            self.metrics["memory_usage"]["peak_mb"] = memory_mb
        self.metrics["memory_usage"]["final_mb"] = memory_mb

    def update_storage_metrics(self, num_nodes):
        """Update storage metrics."""
        if os.path.exists(TEST_FILE):
            file_size_bytes = os.path.getsize(TEST_FILE)
            self.metrics["storage"]["file_size_mb"] = file_size_bytes / 1024 / 1024
            self.metrics["storage"]["bytes_per_node"] = file_size_bytes / max(
                1, num_nodes
            )

        index_file = f"{TEST_FILE}map"
        if os.path.exists(index_file):
            self.metrics["storage"]["index_size_mb"] = (
                os.path.getsize(index_file) / 1024 / 1024
            )

    def record_operation(self, operation, duration, count=1, results=None):
        """Record an operation's performance metrics."""
        if operation in self.metrics:
            self.metrics[operation]["total_time"] += duration
            self.metrics[operation]["operations"] += count

            if operation in ["insert", "retrieval"]:
                self.metrics[operation]["nodes_per_second"] = (
                    count / duration if duration > 0 else 0
                )
            elif operation in ["content_recall", "tag_recall", "spreading_activation"]:
                self.metrics[operation]["avg_time_ms"] = (
                    (duration * 1000) / count if count > 0 else 0
                )
                if results is not None:
                    self.metrics[operation]["avg_results"] += (
                        len(results) / count if count > 0 else 0
                    )

    def print_report(self):
        """Print performance report."""
        logger.info("=" * 80)
        logger.info("EXTREME PERFORMANCE TEST RESULTS")
        logger.info("=" * 80)

        # Insertion performance
        if self.metrics["insert"]["operations"] > 0:
            logger.info(f"Insertion Performance:")
            logger.info(f"  - Total nodes: {self.metrics['insert']['operations']:,}")
            logger.info(
                f"  - Total time: {self.metrics['insert']['total_time']:.2f} seconds"
            )
            logger.info(
                f"  - Speed: {self.metrics['insert']['nodes_per_second']:.2f} nodes/second"
            )

        # Retrieval performance
        if self.metrics["retrieval"]["operations"] > 0:
            logger.info(f"Retrieval Performance:")
            logger.info(
                f"  - Total operations: {self.metrics['retrieval']['operations']:,}"
            )
            logger.info(
                f"  - Total time: {self.metrics['retrieval']['total_time']:.2f} seconds"
            )
            logger.info(
                f"  - Speed: {self.metrics['retrieval']['nodes_per_second']:.2f} nodes/second"
            )

        # Recall performance
        for recall_type in ["content_recall", "tag_recall", "spreading_activation"]:
            if self.metrics[recall_type]["operations"] > 0:
                logger.info(f"{recall_type.replace('_', ' ').title()} Performance:")
                logger.info(
                    f"  - Operations: {self.metrics[recall_type]['operations']}"
                )
                logger.info(
                    f"  - Average time: {self.metrics[recall_type]['avg_time_ms']:.2f} ms"
                )
                logger.info(
                    f"  - Average results: {self.metrics[recall_type]['avg_results']:.2f}"
                )

        # Memory usage
        logger.info(f"Memory Usage:")
        logger.info(f"  - Peak: {self.metrics['memory_usage']['peak_mb']:.2f} MB")
        logger.info(f"  - Final: {self.metrics['memory_usage']['final_mb']:.2f} MB")

        # Storage metrics
        logger.info(f"Storage Metrics:")
        logger.info(
            f"  - Data file size: {self.metrics['storage']['file_size_mb']:.2f} MB"
        )
        logger.info(
            f"  - Index file size: {self.metrics['storage']['index_size_mb']:.2f} MB"
        )
        logger.info(
            f"  - Average bytes per node: {self.metrics['storage']['bytes_per_node']:.2f}"
        )

        # Concurrency performance
        if self.metrics["concurrent"]["sequential_time"] > 0:
            logger.info(f"Concurrency Performance:")
            logger.info(
                f"  - Sequential time: {self.metrics['concurrent']['sequential_time']:.2f} seconds"
            )
            logger.info(
                f"  - Parallel time: {self.metrics['concurrent']['parallel_time']:.2f} seconds"
            )
            logger.info(f"  - Speedup: {self.metrics['concurrent']['speedup']:.2f}x")

        # Save metrics to JSON
        try:
            with open(f"{TEST_DIR}/metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved to {TEST_DIR}/metrics.json")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

        logger.info("=" * 80)


class TestDataGenerator:
    """Generates test data for extreme performance testing."""

    def __init__(self):
        self.domains = [
            "artificial_intelligence",
            "computer_science",
            "biology",
            "physics",
            "chemistry",
            "mathematics",
            "literature",
            "history",
            "psychology",
            "philosophy",
        ]

        self.domain_topics = {}
        self.all_topics = []
        self.domain_nodes = {}  # domain -> list of node IDs
        self.topic_nodes = {}  # topic -> list of node IDs

        # Initialize topics for each domain
        for domain in self.domains:
            self.domain_topics[domain] = [
                f"{domain}_{i+1}" for i in range(TOPICS_PER_DOMAIN)
            ]
            self.domain_nodes[domain] = []
            self.all_topics.extend(self.domain_topics[domain])        # Initialize topic nodes
        for topic in self.all_topics:
            self.topic_nodes[topic] = []

        # Memory decay parameters
        self.base_decay_rates = {
            "working": 0.15,      # Working memory decays very quickly (high decay rate)
            "episodic": 0.08,     # Episodic memories decay moderately fast
            "semantic": 0.02,     # Semantic memory decays slowly (well-established knowledge)
            "procedural": 0.01,   # Procedural memory decays very slowly (motor skills)
            "concept": 0.03,      # Concepts decay slowly            
            "fact": 0.025,        # Facts decay moderately slowly
            "opinion": 0.06,      # Opinions decay faster (more volatile)
            "reference": 0.02,    # References decay slowly
            "note": 0.05,         # Notes decay moderately
            "experience": 0.07    # Experiences decay moderately fast
        }

        # Age-based decay multipliers (older memories decay more)
        self.age_decay_multipliers = {
            "very_recent": 0.5,   # 0-1000 nodes: decay slower (fresh memories)
            "recent": 0.7,        # 1001-3000 nodes: moderate decay
            "moderate": 1.0,      # 3001-6000 nodes: normal decay
            "old": 1.3,           # 6001-8000 nodes: faster decay
            "very_old": 1.6       # 8001+ nodes: much faster decay
        }

    def calculate_memory_level(self, index):
        """Calculate memory level based on creation order (older = higher level)."""
        if index < 1000:
            return "very_recent"
        elif index < 3000:
            return "recent"  
        elif index < 6000:
            return "moderate"
        elif index < 8000:
            return "old"
        else:
            return "very_old"

    def calculate_decay_rate(self, node_type, index):
        """Calculate decay rate based on memory type and age level."""
        base_rate = self.base_decay_rates.get(node_type, 0.04)  # Default decay rate
        age_level = self.calculate_memory_level(index)
        age_multiplier = self.age_decay_multipliers[age_level]
        
        # Calculate final decay rate
        final_decay_rate = base_rate * age_multiplier
        
        # Ensure decay rate stays within reasonable bounds
        return min(max(final_decay_rate, 0.005), 0.25)

    def calculate_initial_activation(self, node_type, index):
        """Calculate initial activation based on memory type and recency."""
        # Base activation levels by memory type
        base_activations = {
            "working": 0.95,      # Working memory starts very active
            "episodic": 0.8,      # Episodic memories start quite active
            "semantic": 0.7,      # Semantic knowledge moderately active
            "procedural": 0.85,   # Procedural skills start active
            "concept": 0.75,      # Concepts moderately active
            "fact": 0.7,          # Facts moderately active
            "opinion": 0.6,       # Opinions less active
            "reference": 0.65,    # References moderately active
            "note": 0.55,         # Notes less active
            "experience": 0.8     # Experiences quite active
        }

        base_activation = base_activations.get(node_type, 0.7)
        age_level = self.calculate_memory_level(index)
        
        # Recent memories start with higher activation
        age_activation_bonus = {
            "very_recent": 0.15,  # +15% for very recent
            "recent": 0.1,        # +10% for recent
            "moderate": 0.0,      # No bonus for moderate
            "old": -0.05,         # -5% for old
            "very_old": -0.1      # -10% for very old
        }        
        
        final_activation = base_activation + age_activation_bonus[age_level]
        return min(max(final_activation, 0.1), 1.0)  # Keep within bounds

    def log_decay_statistics(self, total_nodes):
        """Log statistics about decay rates and activation levels."""
        logger.info("=" * 60)
        logger.info("MEMORY DECAY STATISTICS")
        logger.info("=" * 60)
        
        # Sample nodes to analyze decay patterns
        sample_size = min(1000, total_nodes)
        decay_by_type = {}
        activation_by_type = {}
        decay_by_level = {}
        activation_by_level = {}
        
        for i in range(0, total_nodes, max(1, total_nodes // sample_size)):
            node_type = self.generate_node_type()
            level = self.calculate_memory_level(i)
            decay_rate = self.calculate_decay_rate(node_type, i)
            activation = self.calculate_initial_activation(node_type, i)
            
            # Collect by type
            if node_type not in decay_by_type:
                decay_by_type[node_type] = []
                activation_by_type[node_type] = []
            decay_by_type[node_type].append(decay_rate)
            activation_by_type[node_type].append(activation)
            
            # Collect by level
            if level not in decay_by_level:
                decay_by_level[level] = []
                activation_by_level[level] = []
            decay_by_level[level].append(decay_rate)
            activation_by_level[level].append(activation)
        
        # Log averages by memory type
        logger.info("Average Decay Rates by Memory Type:")
        for node_type in sorted(decay_by_type.keys()):
            avg_decay = sum(decay_by_type[node_type]) / len(decay_by_type[node_type])
            avg_activation = sum(activation_by_type[node_type]) / len(activation_by_type[node_type])
            logger.info(f"  {node_type:12}: Decay={avg_decay:.4f}, Activation={avg_activation:.3f}")
        
        logger.info("\nAverage Decay Rates by Memory Level (Age):")
        level_order = ["very_recent", "recent", "moderate", "old", "very_old"]
        for level in level_order:
            if level in decay_by_level:
                avg_decay = sum(decay_by_level[level]) / len(decay_by_level[level])
                avg_activation = sum(activation_by_level[level]) / len(activation_by_level[level])
                logger.info(f"  {level:12}: Decay={avg_decay:.4f}, Activation={avg_activation:.3f}")
        
        logger.info("=" * 60)

    def generate_content(self, domain, topic, index, node_type=None):
        """Generate content for a node, specialized by memory type."""
        if node_type == "working":
            # Working memory content - temporary, actively processed information
            working_contents = [
                f"Currently processing {domain} information about {topic} item #{index}. This information is being held temporarily for immediate cognitive tasks and will decay quickly without rehearsal.",
                f"Active manipulation of {domain} data from {topic} #{index}. This working memory item contains information being consciously processed and maintained for ongoing cognitive operations.",
                f"Temporary storage of {domain} concepts related to {topic} #{index}. This content represents information in active use for problem-solving and decision-making processes.",
                f"Short-term retention of {domain} knowledge about {topic} #{index}. This working memory content is currently being rehearsed and manipulated for cognitive tasks.",
                f"Immediate processing buffer for {domain} information on {topic} #{index}. This represents the limited-capacity workspace for conscious thought and reasoning.",
            ]
            return random.choice(working_contents)

        elif node_type == "semantic":
            # Semantic memory content - general knowledge and facts
            semantic_contents = [
                f"General knowledge about {domain} in the context of {topic} #{index}. This semantic memory contains factual information and conceptual understanding that persists over time.",
                f"Conceptual understanding of {domain} principles related to {topic} #{index}. This represents well-established knowledge that forms the foundation for further learning.",
                f"Factual information about {domain} and {topic} #{index}. This semantic memory content includes definitions, relationships, and categorical knowledge.",
                f"Abstract knowledge representation of {domain} concepts within {topic} #{index}. This semantic content provides the conceptual framework for understanding the domain.",
                f"Structured knowledge about {domain} regarding {topic} #{index}. This semantic memory contains organized factual information and conceptual relationships.",
            ]
            return random.choice(semantic_contents)

        elif node_type == "episodic":
            # Episodic memory content - specific experiences and events
            episodic_contents = [
                f"Personal experience with {domain} while studying {topic} #{index}. This episodic memory captures a specific learning event with contextual details.",
                f"Remembered encounter with {domain} concepts during {topic} #{index} session. This contains autobiographical details and emotional context.",
                f"Specific learning episode about {domain} topic {topic} #{index}. This episodic memory includes temporal and spatial context of the experience.",
                f"Autobiographical memory of discovering {domain} insights in {topic} #{index}. This contains personal narrative and experiential details.",
            ]
            return random.choice(episodic_contents)

        elif node_type == "procedural":
            # Procedural memory content - skills and procedures
            procedural_contents = [
                f"Step-by-step procedure for {domain} tasks in {topic} #{index}. This procedural memory contains motor and cognitive skills that operate automatically.",
                f"Learned methodology for {domain} operations related to {topic} #{index}. This represents skill-based knowledge and habitual responses.",
                f"Automated process for handling {domain} problems in {topic} #{index}. This procedural content captures learned motor and cognitive routines.",
            ]
            return random.choice(procedural_contents)

        else:
            # Default content for other memory types
            contents = [
                f"This is {domain} content about {topic} item #{index}. It contains detailed information and connects to related concepts in the field.",
                f"An important concept in {domain} related to {topic} is concept #{index}, which explores the fundamental principles and applications.",
                f"Research in {domain} on {topic} item #{index} has shown significant advancements and practical implications for real-world scenarios.",
                f"The relationship between {domain} and {topic} item #{index} demonstrates the interconnected nature of knowledge across different domains.",
            ]
            return random.choice(contents)

    def generate_node_type(self):
        """Generate a random node type including working and semantic memory types."""
        types = [
            "concept",
            "fact",
            "opinion",
            "reference",
            "note",
            "experience",
            "working",
            "semantic",
            "episodic",
            "procedural",
        ]
        weights = [
            0.25,
            0.30,
            0.08,
            0.04,
            0.04,
            0.08,
            0.10,
            0.08,
            0.02,
            0.01,
        ]  # Include working and semantic memory
        return random.choices(types, weights=weights, k=1)[0]

    def generate_tags(self, domain, topic):
        """Generate tags for a node."""
        general_tags = [
            "research",
            "important",
            "review",
            "core",
            "applied",
            "theoretical",
            "experimental",
        ]
        num_tags = random.randint(2, MAX_TAGS_PER_NODE)

        tags = [domain, topic]
        remaining_tags = num_tags - 2
        if remaining_tags > 0:
            tags.extend(
                random.sample(general_tags, min(remaining_tags, len(general_tags)))
            )

        return tags[:MAX_TAGS_PER_NODE]  # Ensure we don't exceed the maximum

    def select_related_nodes(self, domain, topic, all_nodes, current_index):
        """Select related nodes for linking."""
        related_ids = []

        # Nodes from same topic have higher probability
        topic_nodes = self.topic_nodes.get(topic, [])
        if topic_nodes and len(topic_nodes) > current_index:
            same_topic_candidates = topic_nodes[
                :current_index
            ]  # Only use already created nodes
            num_same_topic = min(random.randint(0, 3), len(same_topic_candidates))
            if num_same_topic > 0:
                related_ids.extend(random.sample(same_topic_candidates, num_same_topic))

        # Nodes from same domain but different topic
        domain_nodes = self.domain_nodes.get(domain, [])
        if domain_nodes:
            # Get domain nodes that aren't already in related_ids
            domain_candidates = [n for n in domain_nodes if n not in related_ids]
            num_domain = min(random.randint(0, 2), len(domain_candidates))
            if num_domain > 0:
                related_ids.extend(random.sample(domain_candidates, num_domain))

        # Occasionally add cross-domain links
        if random.random() < 0.1 and len(all_nodes) > 20:
            cross_domain_candidates = [n for n in all_nodes if n not in related_ids]
            num_cross = min(random.randint(0, 2), len(cross_domain_candidates))
            if num_cross > 0:
                related_ids.extend(random.sample(cross_domain_candidates, num_cross))

        # Limit total links
        if len(related_ids) > MAX_LINKS_PER_NODE:
            related_ids = random.sample(related_ids, MAX_LINKS_PER_NODE)

        return related_ids


def generate_test_nodes(engine, generator, metrics):
    """Generate test nodes for extreme performance testing."""
    logger.info(f"Generating {NUM_NODES:,} test nodes across {NUM_DOMAINS} domains...")

    all_node_ids = []
    batch_insert_times = []

    # Process domains in a round-robin fashion to create a balanced dataset
    domains_cycle = generator.domains.copy()
    domain_index = 0

    for batch_start in range(0, NUM_NODES, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, NUM_NODES)
        batch_size = batch_end - batch_start

        logger.info(
            f"Inserting batch {batch_start//BATCH_SIZE + 1}/{(NUM_NODES+BATCH_SIZE-1)//BATCH_SIZE}: nodes {batch_start}-{batch_end-1}"
        )

        start_time = time.time()
        batch_node_ids = []

        for i in range(batch_size):
            # Select domain and topic in a round-robin fashion
            domain = domains_cycle[domain_index % len(domains_cycle)]
            domain_index += 1

            # Select a random topic from the domain
            topic = random.choice(generator.domain_topics[domain])

            # Generate node data
            node_type = generator.generate_node_type()
            content = generator.generate_content(
                domain, topic, batch_start + i, node_type
            )
            tags = generator.generate_tags(domain, topic)            # Select related nodes
            related_ids = generator.select_related_nodes(
                domain, topic, all_node_ids, batch_start + i
            )

            # Calculate decay rate and initial activation based on memory type and age
            decay_rate = generator.calculate_decay_rate(node_type, batch_start + i)
            initial_activation = generator.calculate_initial_activation(node_type, batch_start + i)
            
            # Insert the memory node with calculated decay parameters
            try:
                node_id = engine.insert_memory(
                    content=content,
                    node_type=node_type,
                    tags=tags,
                    related_ids=related_ids,
                    activation=initial_activation,
                    decay_rate=decay_rate,
                )

                # Store node ID
                batch_node_ids.append(node_id)
                all_node_ids.append(node_id)

                # Update domain and topic tracking
                generator.domain_nodes[domain].append(node_id)
                generator.topic_nodes[topic].append(node_id)

            except Exception as e:
                logger.error(f"Failed to insert node {batch_start + i}: {e}")
                # Try without decay parameters in case the engine doesn't support them
                try:
                    node_id = engine.insert_memory(
                        content=content,
                        node_type=node_type,
                        tags=tags,
                        related_ids=related_ids,
                    )
                    batch_node_ids.append(node_id)
                    all_node_ids.append(node_id)
                    generator.domain_nodes[domain].append(node_id)
                    generator.topic_nodes[topic].append(node_id)
                except Exception as e2:
                    logger.error(f"Failed to insert node {batch_start + i} even without decay params: {e2}")

        # Record batch insertion time
        end_time = time.time()
        batch_time = end_time - start_time
        batch_insert_times.append(batch_time)

        # Update metrics
        metrics.record_operation("insert", batch_time, batch_size)
        metrics.update_memory_usage()        # Log batch completion
        nodes_per_second = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            f"Inserted {batch_size} nodes in {batch_time:.2f}s ({nodes_per_second:.2f} nodes/sec)"
        )
        
        # Log decay information for a sample of nodes
        if batch_start % (BATCH_SIZE * 5) == 0:  # Every 5 batches
            sample_node_idx = batch_start + random.randint(0, batch_size - 1)
            sample_type = generator.generate_node_type()
            sample_decay = generator.calculate_decay_rate(sample_type, sample_node_idx)
            sample_activation = generator.calculate_initial_activation(sample_type, sample_node_idx)
            memory_level = generator.calculate_memory_level(sample_node_idx)
            logger.info(
                f"Sample decay info - Node #{sample_node_idx}, Type: {sample_type}, "
                f"Level: {memory_level}, Decay Rate: {sample_decay:.4f}, "
                f"Initial Activation: {sample_activation:.3f}"
            )

        # Force garbage collection to minimize memory growth
        if batch_start % (BATCH_SIZE * 10) == 0:
            gc.collect()

    # Update storage metrics
    metrics.update_storage_metrics(len(all_node_ids))

    logger.info(f"Successfully inserted {len(all_node_ids):,} nodes")
    return all_node_ids


def generate_test_nodes_batch(engine, generator, metrics):
    """Generate test nodes using batch insertion for extreme performance testing."""
    logger.info(
        f"Generating {NUM_NODES:,} test nodes across {NUM_DOMAINS} domains (batch mode)..."
    )

    all_node_ids = []
    batch_insert_times = []

    # Process domains in a round-robin fashion to create a balanced dataset
    domains_cycle = generator.domains.copy()
    domain_index = 0

    # Use a queue to manage batches
    batch_queue = queue.Queue()

    for batch_start in range(0, NUM_NODES, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, NUM_NODES)
        batch_size = batch_end - batch_start

        logger.info(
            f"Preparing batch {batch_start//BATCH_SIZE + 1}/{(NUM_NODES+BATCH_SIZE-1)//BATCH_SIZE}: nodes {batch_start}-{batch_end-1}"
        )

        batch_data = []

        for i in range(batch_size):
            # Select domain and topic in a round-robin fashion
            domain = domains_cycle[domain_index % len(domains_cycle)]
            domain_index += 1

            # Select a random topic from the domain
            topic = random.choice(generator.domain_topics[domain])  # Generate node data
            node_type = generator.generate_node_type()
            content = generator.generate_content(
                domain, topic, batch_start + i, node_type
            )
            tags = generator.generate_tags(domain, topic)

            # Select related nodes
            related_ids = generator.select_related_nodes(
                domain, topic, all_node_ids, batch_start + i
            )

            # Prepare the memory node data for batch insertion
            batch_data.append(
                {
                    "content": content,
                    "node_type": node_type,
                    "tags": tags,
                    "related_ids": related_ids,
                }
            )

        # Add batch to queue
        batch_queue.put(batch_data)

    # Define a worker function for processing batches
    def process_batch(worker_id, queue, metrics):
        logger.info(f"Worker {worker_id} starting...")
        while not queue.empty():
            try:
                # Get the next batch from the queue
                batch_data = queue.get(timeout=1)

                # Insert the batch of memory nodes
                start_time = time.time()
                node_ids = engine.insert_memory_batch(batch_data)
                end_time = time.time()

                # Record batch insertion time
                batch_time = end_time - start_time
                batch_insert_times.append(batch_time)

                # Update metrics
                metrics.record_operation("insert", batch_time, len(node_ids))
                metrics.update_memory_usage()

                # Log batch completion
                logger.info(
                    f"Worker {worker_id} inserted batch of {len(node_ids)} nodes in {batch_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} finished")

    # Create worker threads for parallel batch processing
    num_workers = min(CONCURRENT_OPERATIONS, NUM_DOMAINS)
    threads = []
    for worker_id in range(num_workers):
        thread = threading.Thread(
            target=process_batch, args=(worker_id, batch_queue, metrics)
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Update storage metrics
    metrics.update_storage_metrics(len(all_node_ids))

    logger.info(f"Successfully inserted {len(all_node_ids):,} nodes (batch mode)")
    return all_node_ids


def generate_high_performance_nodes(engine, generator, metrics):
    """
    Generate test nodes using optimized RAM-first batch insertion
    with deferred disk writes for extreme performance.

    This implements:
    1. RAM-first operations with batched disk writes
    2. One background writer thread to avoid file locking issues
    3. Serialization batching to reduce overhead
    4. Deferred disk flushing to minimize I/O bottlenecks
    """
    logger.info(
        f"Generating {NUM_NODES:,} test nodes across {NUM_DOMAINS} domains (high performance mode)..."
    )

    all_node_ids = []
    batch_insert_times = []

    # Get direct access to RAMX for fastest insertion
    ramx = None
    if hasattr(engine.storage, "ramx"):
        ramx = engine.storage.ramx
    else:
        logger.warning(
            "Direct RAMX access not available, falling back to standard insertion"
        )
        return generate_test_nodes(engine, generator, metrics)

    # Create writer queue for batching disk operations
    writer_queue = queue.Queue()
    shutdown_flag = threading.Event()

    # Create background writer thread
    def background_writer():
        """Background thread for batched disk writes"""
        logger.info("Background writer thread started")
        batch_counter = 0

        while not shutdown_flag.is_set() or not writer_queue.empty():
            try:
                # Wait for a batch of nodes or timeout
                try:
                    batch = writer_queue.get(timeout=0.5)
                    batch_counter += 1
                except queue.Empty:
                    continue
                # Get orchestrator - either directly or through adapter
                orchestrator = None
                if hasattr(engine.storage, "orchestrator"):
                    # If storage is an adapter
                    orchestrator = engine.storage.orchestrator
                elif hasattr(engine.storage, "serializer"):
                    # If storage is already an orchestrator
                    orchestrator = engine.storage
                else:
                    logger.error("No orchestrator found, cannot write batch to disk")
                    continue

                # Process the batch of nodes
                start_time = time.time()

                # Batch serialize all nodes at once
                serialized_batch = []
                for node in batch:
                    try:
                        serialized = orchestrator.serialize_to_bytes(node)
                        serialized_batch.append((node.id, serialized))
                    except Exception as e:
                        logger.error(f"Failed to serialize node {node.id}: {e}")

                # Acquire a single file lock for the entire batch
                try:
                    # Open data file for append
                    with open(orchestrator.data_path, "ab") as f:
                        # Write all nodes in one file operation
                        for node_id, serialized_data in serialized_batch:
                            # Get current position (offset)
                            offset = f.tell()

                            # Write the serialized data
                            f.write(serialized_data)

                            # Update index with offset
                            orchestrator.index_manager.update_index(
                                orchestrator.index_path, node_id, offset
                            )

                        # Ensure data is on disk
                        f.flush()
                        os.fsync(f.fileno())

                    end_time = time.time()
                    write_time = end_time - start_time

                    logger.debug(
                        f"Batch {batch_counter}: Wrote {len(batch)} nodes to disk in {write_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Failed to write batch to disk: {e}")

            except Exception as e:
                logger.error(f"Error in background writer: {e}")

        logger.info("Background writer thread finished")

    # Start background writer thread
    writer_thread = threading.Thread(target=background_writer)
    writer_thread.daemon = True
    writer_thread.start()

    try:
        # Process domains in a round-robin fashion to create a balanced dataset
        domains_cycle = generator.domains.copy()
        domain_index = 0

        # Buffer for nodes awaiting disk write
        node_buffer = []

        for batch_start in range(0, NUM_NODES, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_NODES)
            batch_size = batch_end - batch_start

            logger.info(
                f"Generating batch {batch_start//BATCH_SIZE + 1}/{(NUM_NODES+BATCH_SIZE-1)//BATCH_SIZE}: nodes {batch_start}-{batch_end-1}"
            )

            start_time = time.time()
            batch_node_ids = []
            batch_nodes = []

            for i in range(batch_size):
                # Select domain and topic in a round-robin fashion
                domain = domains_cycle[domain_index % len(domains_cycle)]
                domain_index += 1

                # Select a random topic from the domain
                topic = random.choice(generator.domain_topics[domain])                # Generate node data
                node_type = generator.generate_node_type()
                content = generator.generate_content(domain, topic, batch_start + i)
                tags = generator.generate_tags(domain, topic)

                # Select related nodes
                related_ids = generator.select_related_nodes(
                    domain, topic, all_node_ids, batch_start + i
                )
                
                # Calculate decay rate and initial activation based on memory type and age
                decay_rate = generator.calculate_decay_rate(node_type, batch_start + i)
                initial_activation = generator.calculate_initial_activation(node_type, batch_start + i)
                
                # Create memory node directly
                node_id = uuid4()

                # Create node (directly using RAMX for best performance)
                ramx_node = RAMXNode(
                    id=node_id,
                    content=content,
                    node_type=node_type,
                    version=1,
                    timestamp=datetime.now().timestamp(),
                    activation=initial_activation,
                    decay_rate=decay_rate,
                    version_of=None,
                    links={},
                    tags=tags,
                )

                # Add links separately
                for rel_id in related_ids:
                    ramx_node.add_link(to_id=rel_id, weight=0.6, link_type="related")

                # Add directly to RAMX (fastest in-memory insertion)
                ramx.add_node(ramx_node)

                # Convert to MemoryNode for disk serialization
                memory_node = ramx_node.to_memory_node()

                # Add to node buffer
                node_buffer.append(memory_node)

                # Store node ID
                batch_node_ids.append(node_id)
                all_node_ids.append(node_id)

                # Update domain and topic tracking
                generator.domain_nodes[domain].append(node_id)
                generator.topic_nodes[topic].append(node_id)

            # Batch disk operations - queue for background writer when buffer size threshold reached
            if len(node_buffer) >= WRITE_BUFFER_SIZE:
                writer_queue.put(node_buffer.copy())
                node_buffer.clear()
                logger.debug(f"Queued {WRITE_BUFFER_SIZE} nodes for background writing")

            # Record batch insertion time
            end_time = time.time()
            batch_time = end_time - start_time
            batch_insert_times.append(batch_time)

            # Update metrics
            metrics.record_operation("insert", batch_time, batch_size)
            metrics.update_memory_usage()

            # Log batch completion
            nodes_per_second = batch_size / batch_time if batch_time > 0 else 0
            logger.info(
                f"Generated batch {batch_start//BATCH_SIZE + 1}: {batch_size} nodes in {batch_time:.2f}s ({nodes_per_second:.2f} nodes/sec)"
            )

            # Force garbage collection to minimize memory growth
            if batch_start % (BATCH_SIZE * 10) == 0:
                gc.collect()

        # Queue any remaining nodes for disk write
        if node_buffer:
            writer_queue.put(node_buffer)
            node_buffer.clear()

    finally:
        # Wait for background writer to finish
        logger.info("Waiting for background writer to complete...")
        shutdown_flag.set()
        writer_thread.join(timeout=30)

        # Force a flush to ensure all data is written
        if hasattr(engine.storage, "flush"):
            engine.storage.flush()

    # Update storage metrics
    metrics.update_storage_metrics(len(all_node_ids))

    logger.info(
        f"Successfully inserted {len(all_node_ids):,} nodes (high-performance mode)"
    )
    return all_node_ids


def generate_ultra_performance_nodes(engine, generator, metrics):
    """
    Generate test nodes using ultra-high-performance RAM-first strategies with
    optimized async disk I/O and minimal serialization.

    This function implements extreme optimizations:
    1. Pure in-memory operations with deferred disk writes
    2. Single shared memory structure to avoid serialization bottlenecks
    3. Timed disk flushing instead of threshold-based to reduce lock contention
    4. Bulk serialization of batches to minimize overhead
    5. File append-only log with single write operation per batch
    6. Minimal disk syncs and reduced fsync calls
    """
    logger.info(
        f"Generating {NUM_NODES:,} test nodes across {NUM_DOMAINS} domains (ULTRA performance mode)..."
    )

    all_node_ids = []
    batch_insert_times = []
    node_batches = []  # Store batches for writing

    # Configure ultra-performance settings
    ULTRA_WRITE_BUFFER_SIZE = BATCH_SIZE * 10  # Much larger buffer for fewer writes
    ULTRA_FLUSH_INTERVAL = (
        10.0  # Seconds between flushes (time-based rather than count)
    )
    ULTRA_MAX_PENDING_BATCHES = (
        5  # Maximum number of batches to queue before forcing flush
    )

    # Get direct access to RAMX for fastest insertion
    ramx = None
    if hasattr(engine.storage, "ramx"):
        ramx = engine.storage.ramx
    else:
        logger.warning(
            "Direct RAMX access not available, falling back to high-performance mode"
        )
        return generate_high_performance_nodes(
            engine, generator, metrics
        )  # Get orchestrator for background disk operations
    orchestrator = None

    # Try to find the orchestrator in multiple ways
    if hasattr(engine.storage, "orchestrator"):
        # If storage is a MemoryStorageAdapter
        orchestrator = engine.storage.orchestrator
        logger.info(f"Found orchestrator via adapter: {type(orchestrator).__name__}")
    elif hasattr(engine.storage, "serializer") and hasattr(engine.storage, "ramx"):
        # If storage is already a MemoryIOOrchestrator (has both serializer and ramx)
        orchestrator = engine.storage
        logger.info(f"Found orchestrator directly: {type(orchestrator).__name__}")
    elif "MemoryIOOrchestrator" in str(type(engine.storage)):
        # String-based class name check as fallback
        orchestrator = engine.storage
        logger.info(f"Found orchestrator via class name: {type(orchestrator).__name__}")

    # Final check if we found the orchestrator
    if orchestrator is None:
        logger.warning(
            f"Orchestrator not available (storage type: {type(engine.storage).__name__}), falling back to high-performance mode"
        )
        return generate_high_performance_nodes(engine, generator, metrics)

    # Create shared memory structures (to minimize copying)
    shared_buffer = []
    pending_nodes = 0
    last_flush_time = time.time()

    # Background writer flags
    writer_queue = queue.Queue()
    shutdown_flag = threading.Event()

    # Batch generation statistics
    batch_generated = 0
    batch_written = 0

    # Create ultra-optimized serialization function for entire batches at once
    def serialize_batch(nodes):
        """
        Serialize an entire batch of nodes at once instead of one-by-one
        """
        # Pre-allocate memory for serialized data
        serialized_batch = []
        start_time = time.time()

        # First pass: collect all serialized data without writing
        for node in nodes:
            try:
                serialized = orchestrator.serialize_to_bytes(node)
                serialized_batch.append((node.id, serialized))
            except Exception as e:
                logger.error(f"Failed to serialize node {node.id}: {e}")

        end_time = time.time()
        serialize_time = end_time - start_time
        logger.debug(
            f"Serialized {len(nodes)} nodes in {serialize_time:.4f}s ({len(nodes)/serialize_time:.2f} nodes/sec)"
        )

        return serialized_batch

    # Create optimized background writer that uses append-only log semantics
    def background_writer():
        """Background thread for optimized, batched disk writes"""
        logger.info("Ultra-performance background writer thread started")
        batch_counter = 0

        # Create a buffer for merging multiple batches into one write operation
        mega_batch = []
        mega_batch_size = 0
        last_write_time = time.time()

        while not shutdown_flag.is_set() or not writer_queue.empty():
            try:
                # Check if we should flush based on time or queue size
                current_time = time.time()
                time_since_last_write = current_time - last_write_time
                should_flush = mega_batch_size > 0 and (
                    time_since_last_write >= ULTRA_FLUSH_INTERVAL
                    or writer_queue.qsize() >= ULTRA_MAX_PENDING_BATCHES
                )

                # Wait for a batch of nodes or timeout
                try:
                    # Use shorter timeout if we have data to flush
                    timeout = 0.1 if mega_batch_size > 0 else 0.5
                    batch = writer_queue.get(timeout=timeout)
                    batch_counter += 1

                    # Add to mega batch
                    mega_batch.extend(batch)
                    mega_batch_size += len(batch)

                except queue.Empty:
                    # If timed out, continue to flush check
                    pass

                # Check if we need to flush the mega batch
                if mega_batch_size > 0 and (should_flush or shutdown_flag.is_set()):
                    start_time = time.time()

                    # Bulk-serialize the entire mega batch at once
                    serialized_items = serialize_batch(mega_batch)

                    try:
                        # Use append-only log strategy - single file open/close
                        with open(orchestrator.data_path, "ab") as f:
                            # Track offsets for all nodes
                            index_updates = {}

                            # Write all nodes in one file operation
                            for node_id, serialized_data in serialized_items:
                                # Get current position (offset)
                                offset = f.tell()
                                index_updates[node_id] = offset

                                # Write the serialized data
                                f.write(serialized_data)

                            # Only flush once per mega batch
                            f.flush()

                            # Only fsync periodically to avoid excessive disk I/O
                            if time_since_last_write >= 5.0 or shutdown_flag.is_set():
                                os.fsync(f.fileno())

                        # Update all indexes in a single batch operation
                        orchestrator.index_manager.batch_update_index(
                            orchestrator.index_path, index_updates
                        )

                        end_time = time.time()
                        write_time = end_time - start_time

                        logger.info(
                            f"ULTRA Writer: Wrote mega-batch of {mega_batch_size} nodes "
                            f"({batch_counter} batches) in {write_time:.2f}s "
                            f"({mega_batch_size/write_time:.2f} nodes/sec)"
                        )

                        # Reset mega batch
                        mega_batch.clear()
                        mega_batch_size = 0
                        last_write_time = current_time

                    except Exception as e:
                        logger.error(f"Failed to write mega-batch to disk: {e}")

            except Exception as e:
                logger.error(f"Error in background writer: {e}")

        logger.info(
            f"Ultra-performance background writer finished (processed {batch_counter} batches)"
        )

    # Start background writer thread
    writer_thread = threading.Thread(target=background_writer)
    writer_thread.daemon = True
    writer_thread.start()

    # Helper function to process worker results
    def _process_worker_result(result_dict, worker_result):
        """Store worker results in the result dictionary"""
        node_ids, batch_times = worker_result
        result_dict["node_ids"] = node_ids
        result_dict["batch_times"] = batch_times

    # Use a multi-threaded approach for node generation
    def generate_nodes_worker(worker_id, start_index, end_index):
        """Worker function to generate nodes in parallel"""
        nonlocal pending_nodes

        local_node_ids = []
        local_batch_times = []
        local_node_buffer = []

        worker_domains_cycle = generator.domains.copy()
        # Offset each worker's starting domain to avoid contention
        worker_domain_index = (
            worker_id * len(worker_domains_cycle) // NUM_WORKERS
        ) % len(worker_domains_cycle)

        logger.info(
            f"ULTRA Worker {worker_id}: Generating nodes {start_index} to {end_index-1}"
        )

        for i in range(start_index, end_index):
            start_time = time.perf_counter() if i % 1000 == 0 else 0

            # Select domain and topic in a round-robin fashion with worker offset
            domain = worker_domains_cycle[
                worker_domain_index % len(worker_domains_cycle)
            ]
            worker_domain_index += 1

            # Select a random topic from the domain
            topic = random.choice(generator.domain_topics[domain])            # Generate node data
            node_type = generator.generate_node_type()
            content = generator.generate_content(domain, topic, i)
            tags = generator.generate_tags(domain, topic)

            # Select related nodes - limit to recent nodes for better performance
            # Use a time-based decay window to prioritize recent nodes and limit search space
            recent_window = min(
                100000, len(all_node_ids)
            )  # Look at most recent 10K nodes
            related_ids = generator.select_related_nodes(
                domain,
                topic,
                all_node_ids[-recent_window:] if recent_window > 0 else [],
                i,
            )

            # Calculate decay rate and initial activation based on memory type and age
            decay_rate = generator.calculate_decay_rate(node_type, i)
            initial_activation = generator.calculate_initial_activation(node_type, i)

            # Create node directly with UUID (avoid going through engine APIs)
            node_id = uuid4()

            # Create RAMX node directly (maximum performance)
            ramx_node = RAMXNode(
                id=node_id,
                content=content,
                node_type=node_type,
                version=1,
                timestamp=datetime.now().timestamp(),
                activation=initial_activation,
                decay_rate=decay_rate,
                version_of=None,
                links={},
                tags=tags,
            )

            # Add links directly for best performance
            for rel_id in related_ids:
                ramx_node.add_link(to_id=rel_id, weight=0.6, link_type="related")

            # Add directly to RAMX
            with threading.Lock():  # Short lock to add to RAMX
                ramx.add_node(ramx_node)

            # Convert to MemoryNode for disk serialization later
            memory_node = ramx_node.to_memory_node()

            # Add to local node buffer
            local_node_buffer.append(memory_node)

            # Store node ID
            local_node_ids.append(node_id)

            # Update domain and topic tracking (thread-safe using local buffers)
            if domain not in generator.domain_nodes:
                generator.domain_nodes[domain] = []
            generator.domain_nodes[domain].append(node_id)

            if topic not in generator.topic_nodes:
                generator.topic_nodes[topic] = []
            generator.topic_nodes[topic].append(node_id)

            # Periodically batch and queue nodes for background writing
            if len(local_node_buffer) >= BATCH_SIZE:
                # Create a copy of the local buffer for the writer
                nodes_to_queue = local_node_buffer.copy()

                with threading.Lock():  # Protect shared queue access
                    writer_queue.put(nodes_to_queue)
                    pending_nodes += len(nodes_to_queue)

                # Measure batch generation time
                if start_time > 0:
                    batch_time = time.perf_counter() - start_time
                    local_batch_times.append(batch_time)
                    nodes_per_second = BATCH_SIZE / batch_time if batch_time > 0 else 0
                    logger.info(
                        f"ULTRA Worker {worker_id}: Generated batch {i//BATCH_SIZE + 1}: "
                        f"{BATCH_SIZE} nodes in {batch_time:.2f}s ({nodes_per_second:.2f} nodes/sec)"
                    )

                # Clear local buffer after queuing
                local_node_buffer.clear()

            # Periodically perform garbage collection in each worker
            if i % (BATCH_SIZE * 20) == 0 and i > 0:
                gc.collect()

        # Queue any remaining nodes
        if local_node_buffer:
            with threading.Lock():  # Protect shared queue access
                writer_queue.put(local_node_buffer.copy())
                pending_nodes += len(local_node_buffer)
            local_node_buffer.clear()

        # Return worker results
        return local_node_ids, local_batch_times

    try:
        # Parallel node generation using multiple threads
        NUM_WORKERS = min(os.cpu_count() or 4, CONCURRENT_OPERATIONS)
        logger.info(f"Using {NUM_WORKERS} parallel generation workers")

        # Calculate work distribution
        nodes_per_worker = NUM_NODES // NUM_WORKERS
        remaining_nodes = NUM_NODES % NUM_WORKERS

        # Create and start worker threads
        worker_threads = []
        worker_results = []

        start_index = 0
        for worker_id in range(NUM_WORKERS):
            # Distribute remaining nodes
            worker_nodes = nodes_per_worker + (1 if worker_id < remaining_nodes else 0)
            end_index = start_index + worker_nodes

            # Store thread result wrapper
            result = {"node_ids": [], "batch_times": []}
            worker_results.append(result)
            # Create and start thread
            thread = threading.Thread(
                target=lambda wid=worker_id, si=start_index, ei=end_index, r=result: _process_worker_result(
                    r, generate_nodes_worker(wid, si, ei)
                )
            )
            thread.start()
            worker_threads.append(thread)

            # Update start index for next worker
            start_index = end_index

        # Wait for all worker threads to complete
        for thread in worker_threads:
            thread.join()

        # Combine results from all workers
        for result in worker_results:
            all_node_ids.extend(result["node_ids"])
            batch_insert_times.extend(result["batch_times"])

        logger.info(
            f"All node generation workers completed, generated {len(all_node_ids)} nodes"
        )

    finally:
        # Wait for background writer to finish
        logger.info(
            f"Waiting for background writer to complete ({writer_queue.qsize()} batches pending)..."
        )
        shutdown_flag.set()
        writer_thread.join(timeout=60)

        # Force a flush to ensure all data is written
        if hasattr(engine.storage, "flush"):
            engine.storage.flush()

    # Update storage metrics
    metrics.update_storage_metrics(len(all_node_ids))

    # Calculate and log performance metrics
    if batch_insert_times:
        avg_time = sum(batch_insert_times) / len(batch_insert_times)
        avg_rate = BATCH_SIZE / avg_time if avg_time > 0 else 0
        logger.info(
            f"Average batch generation time: {avg_time:.2f}s ({avg_rate:.2f} nodes/sec)"
        )

    logger.info(
        f"Successfully inserted {len(all_node_ids):,} nodes (ULTRA-performance mode)"
    )
    return all_node_ids


def test_retrieval_performance(engine, node_ids, metrics, sample_size=1000):
    """Test retrieval performance for a sample of nodes."""
    logger.info(f"Testing retrieval performance with {sample_size} random nodes...")

    # Select random nodes to test retrieval
    test_nodes = random.sample(node_ids, min(sample_size, len(node_ids)))

    start_time = time.time()
    successful_retrievals = 0

    for node_id in test_nodes:
        try:
            node = engine.get_memory(node_id)
            if node:
                successful_retrievals += 1
        except Exception as e:
            logger.error(f"Error retrieving node {node_id}: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    metrics.record_operation("retrieval", total_time, successful_retrievals)

    retrieval_rate = successful_retrievals / len(test_nodes) * 100
    nodes_per_second = successful_retrievals / total_time if total_time > 0 else 0

    logger.info(
        f"Retrieved {successful_retrievals}/{len(test_nodes)} nodes ({retrieval_rate:.2f}%) in {total_time:.2f}s"
    )
    logger.info(f"Retrieval performance: {nodes_per_second:.2f} nodes/second")

    return successful_retrievals


def test_content_recall(engine, metrics, num_queries=50):
    """Test content-based recall performance."""
    logger.info(
        f"Testing content-based recall with {num_queries} queries..."
    )  # Generate a variety of queries
    queries = [
        "artificial intelligence machine learning",
        "computer science algorithms data structures",
        "biology genetics dna",
        "physics quantum mechanics",
        "mathematics probability statistics",
        "psychology cognitive behavioral",
        "literature analysis interpretation",
        "history ancient civilizations",
        "philosophy ethics morality",
        "chemistry organic compounds",
        "working memory temporary",
        "semantic knowledge facts",
        "episodic personal experience",
        "procedural skills automation",
    ]

    # Extend with more specific queries
    specific_terms = [
        "neural networks",
        "deep learning",
        "reinforcement learning",
        "graph theory",
        "complexity analysis",
        "compiler design",
        "gene expression",
        "cellular biology",
        "evolutionary theory",
        "quantum field theory",
        "relativity",
        "string theory",
        "calculus",
        "linear algebra",
        "number theory",
        "developmental psychology",
        "behaviorism",
        "neuroscience",
        "poetry analysis",
        "narrative structure",
        "literary criticism",
        "world war",
        "renaissance",
        "industrial revolution",
        "existentialism",
        "metaphysics",
        "working memory processing",
        "semantic conceptual framework",
        "episodic autobiographical",
        "procedural motor skills",
        "epistemology",
        "thermodynamics",
        "polymer science",
        "biochemistry",
    ]

    for term in specific_terms:
        queries.append(term)

    # Ensure we have enough queries
    while len(queries) < num_queries:
        # Combine existing queries to create new ones
        q1 = random.choice(queries)
        q2 = random.choice(queries)
        queries.append(f"{q1} {q2}")

    # Select a subset of queries to use
    selected_queries = random.sample(queries, min(num_queries, len(queries)))

    total_results = 0
    for query in selected_queries:
        start_time = time.time()

        try:
            # Perform content-based recall
            results = engine.recall_memories(query=query, limit=10)

            end_time = time.time()
            query_time = end_time - start_time

            # Record metrics
            metrics.record_operation("content_recall", query_time, 1, results)

            logger.info(
                f"Query: '{query}' - found {len(results)} results in {query_time*1000:.2f}ms"
            )
            total_results += len(results)

        except Exception as e:
            logger.error(
                f"Error in content recall for query '{query}': {e} {traceback.print_exc()}"
            )
            total_results += 0

    avg_results = total_results / len(selected_queries) if selected_queries else 0
    logger.info(f"Content recall: average {avg_results:.2f} results per query")

    return total_results


def test_tag_recall(engine, metrics, num_queries=50):
    """Test tag-based recall performance."""
    logger.info(f"Testing tag-based recall with {num_queries} queries...")

    # Generate a variety of tag combinations
    domain_tags = [
        "artificial_intelligence",
        "computer_science",
        "biology",
        "physics",
        "chemistry",
        "mathematics",
        "literature",
        "history",
        "psychology",
        "philosophy",
    ]

    general_tags = [
        "research",
        "important",
        "review",
        "core",
        "applied",
        "theoretical",
        "experimental",
    ]

    tag_queries = []

    # Single domain tags
    tag_queries.extend([{"tags": [tag]} for tag in domain_tags])

    # Domain + general tag combinations
    for domain in domain_tags:
        for general in general_tags:
            tag_queries.append({"tags": [domain, general]})

    # Ensure we have enough queries
    while len(tag_queries) < num_queries:
        # Create random tag combinations
        num_tags = random.randint(1, 3)
        tags = random.sample(domain_tags + general_tags, num_tags)
        tag_queries.append({"tags": tags})

    # Select a subset of tag queries to use
    selected_queries = random.sample(tag_queries, min(num_queries, len(tag_queries)))

    total_results = 0
    for query in selected_queries:
        start_time = time.time()

        try:
            # Perform tag-based recall
            results = engine.recall_memories(**query, limit=10)

            end_time = time.time()
            query_time = end_time - start_time

            # Record metrics
            metrics.record_operation("tag_recall", query_time, 1, results)

            logger.info(
                f"Tags: {query['tags']} - found {len(results)} results in {query_time*1000:.2f}ms"
            )
            total_results += len(results)

        except Exception as e:
            logger.error(f"Error in tag recall for tags {query['tags']}: {e}")

    avg_results = total_results / len(selected_queries) if selected_queries else 0
    logger.info(f"Tag recall: average {avg_results:.2f} results per query")

    return total_results


def test_spreading_activation(engine, node_ids, metrics, num_starts=20):
    """Test spreading activation performance."""
    logger.info(f"Testing spreading activation with {num_starts} starting nodes...")

    # Select random starting nodes
    start_nodes = random.sample(node_ids, min(num_starts, len(node_ids)))

    total_results = 0
    for node_id in start_nodes:
        for depth in [1, 2, 3]:
            start_time = time.time()

            try:
                # Perform spreading activation
                results = engine.find_related_memories(node_id, max_depth=depth)

                end_time = time.time()
                query_time = end_time - start_time

                # Record metrics
                metrics.record_operation("spreading_activation", query_time, 1, results)

                logger.info(
                    f"Node {node_id}, depth {depth} - found {len(results)} related nodes in {query_time*1000:.2f}ms"
                )
                total_results += len(results)

            except Exception as e:
                logger.error(
                    f"Error in spreading activation for node {node_id}, depth {depth}: {e}"
                )

    avg_results = total_results / (len(start_nodes) * 3) if start_nodes else 0
    logger.info(f"Spreading activation: average {avg_results:.2f} results per query")

    return total_results


def test_concurrent_operations(engine, node_ids, metrics):
    """Test concurrent operations performance."""
    logger.info(
        f"Testing concurrent operations performance with {CONCURRENT_OPERATIONS} workers..."
    )

    # Define test operations
    def operation_get_memory(node_id):
        return engine.get_memory(node_id)

    def operation_recall_content(query):
        return engine.recall_memories(query=query, limit=5)

    def operation_recall_tags(tags):
        return engine.recall_memories(tags=tags, limit=5)

    def operation_related(node_id):
        return engine.find_related_memories(node_id, max_depth=2)

    # Create a mix of operations
    operations = []

    # Add get_memory operations
    for _ in range(200):
        node_id = random.choice(node_ids)
        operations.append((operation_get_memory, (node_id,)))

    # Add content recall operations
    queries = [
        "artificial intelligence",
        "computer science",
        "biology",
        "physics",
        "mathematics",
        "literature",
        "history",
        "psychology",
        "philosophy",
    ]
    for _ in range(20):
        query = random.choice(queries)
        operations.append((operation_recall_content, (query,)))

    # Add tag recall operations
    domain_tags = [
        "artificial_intelligence",
        "computer_science",
        "biology",
        "physics",
        "chemistry",
        "mathematics",
        "literature",
        "history",
        "psychology",
        "philosophy",
    ]
    for _ in range(20):
        tags = [random.choice(domain_tags)]
        operations.append((operation_recall_tags, (tags,)))

    # Add related memory operations
    for _ in range(20):
        node_id = random.choice(node_ids)
        operations.append((operation_related, (node_id,)))

    # Shuffle operations
    random.shuffle(operations)

    # First run sequentially
    logger.info("Running operations sequentially...")
    sequential_start = time.time()
    sequential_results = []

    for op_func, op_args in operations:
        try:
            result = op_func(*op_args)
            sequential_results.append(result)
        except Exception as e:
            logger.error(f"Error in sequential operation: {e}")
            sequential_results.append(None)

    sequential_time = time.time() - sequential_start
    logger.info(f"Sequential execution completed in {sequential_time:.2f}s")

    # Then run in parallel
    logger.info(f"Running operations with {CONCURRENT_OPERATIONS} parallel workers...")
    parallel_start = time.time()
    parallel_results = [None] * len(operations)

    with ThreadPoolExecutor(max_workers=CONCURRENT_OPERATIONS) as executor:
        futures = []
        for i, (op_func, op_args) in enumerate(operations):
            future = executor.submit(op_func, *op_args)
            futures.append((future, i))

        for future, i in futures:
            try:
                result = future.result()
                parallel_results[i] = result
            except Exception as e:
                logger.error(f"Error in parallel operation: {e}")
                parallel_results[i] = None

    parallel_time = time.time() - parallel_start
    logger.info(f"Parallel execution completed in {parallel_time:.2f}s")

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"Parallel speedup: {speedup:.2f}x")

    # Update metrics
    metrics.metrics["concurrent"]["sequential_time"] = sequential_time
    metrics.metrics["concurrent"]["parallel_time"] = parallel_time
    metrics.metrics["concurrent"]["speedup"] = speedup

    return speedup


def main():
    """Run extreme performance test for enhanced merX system."""
    logger.info("=" * 80)
    logger.info("STARTING EXTREME PERFORMANCE TEST")
    logger.info("=" * 80)

    # Create output directory
    os.makedirs(TEST_DIR, exist_ok=True)

    # Initialize metrics
    metrics = PerformanceMetrics()

    # CPU and system info
    cpu_count = multiprocessing.cpu_count()
    memory_total = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
    logger.info(f"System: {cpu_count} CPU cores, {memory_total:.2f} MB RAM")

    # Create enhanced memory engine with factory
    logger.info("Creating enhanced memory engine...")
    factory = EnhancedMemoryEngineFactory()
    engine = factory.create_engine(
        ram_capacity=NUM_NODES * 2,  # Extra capacity for safety
        data_path=TEST_FILE,
        flush_interval=5.0,
        flush_threshold=1000,
    )

    try:
        # Initialize test data generator
        generator = TestDataGenerator()

        # Generate test nodes
        start_time = time.time()
        node_ids = generate_ultra_performance_nodes(engine, generator, metrics)
        total_insert_time = time.time() - start_time
        logger.info(f"Total insertion time: {total_insert_time:.2f}s")

        # Test retrieval performance
        test_retrieval_performance(engine, node_ids, metrics)

        # Test content recall
        test_content_recall(engine, metrics)

        # Test tag recall
        test_tag_recall(engine, metrics)

        # Test spreading activation
        test_spreading_activation(engine, node_ids, metrics)

        # Test concurrent operations
        test_concurrent_operations(engine, node_ids, metrics)

        # Get memory stats
        stats = engine.get_memory_stats()
        logger.info(f"Memory engine stats: {stats}")

        # Update final memory usage
        metrics.update_memory_usage()

        # Print performance report
        metrics.print_report()

    finally:
        # Ensure proper cleanup
        logger.info("Shutting down enhanced memory engine...")
        factory.cleanup_and_shutdown(engine)

    logger.info("=" * 80)
    logger.info("EXTREME PERFORMANCE TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
