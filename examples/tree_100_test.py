#!/usr/bin/env python3
"""
Tree-of-Trees Test Data Generator - 100 Nodes

This script creates a focused tree-of-trees memory structure with exactly 100 nodes
across 5 knowledge domains, designed for optimal visualization and analysis of
hierarchical relationships in the merX memory system.
"""

import os
import sys
import json
import time
import random
import logging
import traceback
from typing import List, Dict, Any, Tuple
from datetime import datetime
from uuid import uuid4

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tree-of-Trees configuration for 100 nodes
NUM_DOMAINS = 5  # Number of knowledge domains (trees)
NODES_PER_DOMAIN = 20  # 20 nodes per domain = 100 total nodes
TREE_DEPTH = 3  # Hierarchical depth within each domain tree
CROSS_REFERENCES = 10  # Links between different domain trees
INTRA_DOMAIN_LINKS = 15  # Additional links within each domain

# Knowledge domains for tree structure
DOMAINS = {
    "artificial_intelligence": {
        "root_topics": ["machine_learning", "neural_networks", "computer_vision"],
        "subtopics": {
            "machine_learning": [
                "supervised_learning",
                "unsupervised_learning",
                "reinforcement_learning",
            ],
            "neural_networks": ["deep_learning", "cnn", "rnn", "transformer"],
            "computer_vision": [
                "image_processing",
                "object_detection",
                "face_recognition",
            ],
        },
        "sample_content": [
            "Machine learning algorithms enable computers to learn patterns from data.",
            "Neural networks consist of interconnected nodes that process information.",
            "Computer vision systems analyze and interpret visual information.",
            "Deep learning uses multi-layer neural networks for complex tasks.",
            "Supervised learning uses labeled data to train predictive models.",
        ],
    },
    "computer_science": {
        "root_topics": ["algorithms", "data_structures", "programming"],
        "subtopics": {
            "algorithms": [
                "sorting",
                "searching",
                "graph_algorithms",
                "dynamic_programming",
            ],
            "data_structures": ["arrays", "trees", "graphs", "hash_tables"],
            "programming": ["languages", "paradigms", "frameworks", "testing"],
        },
        "sample_content": [
            "Algorithms are step-by-step procedures for solving computational problems.",
            "Data structures organize and store data for efficient access and modification.",
            "Programming involves creating instructions for computers to execute.",
            "Sorting algorithms arrange data elements in a specific order.",
            "Trees are hierarchical data structures with parent-child relationships.",
        ],
    },
    "mathematics": {
        "root_topics": ["calculus", "algebra", "statistics"],
        "subtopics": {
            "calculus": ["derivatives", "integrals", "limits", "optimization"],
            "algebra": ["linear_algebra", "abstract_algebra", "matrices", "vectors"],
            "statistics": [
                "probability",
                "distributions",
                "hypothesis_testing",
                "regression",
            ],
        },
        "sample_content": [
            "Calculus deals with rates of change and accumulation of quantities.",
            "Algebra studies mathematical symbols and rules for manipulating them.",
            "Statistics involves collecting, analyzing, and interpreting data.",
            "Derivatives measure how functions change with respect to variables.",
            "Linear algebra studies vector spaces and linear transformations.",
        ],
    },
    "physics": {
        "root_topics": ["mechanics", "thermodynamics", "electromagnetism"],
        "subtopics": {
            "mechanics": ["kinematics", "dynamics", "statics", "fluid_mechanics"],
            "thermodynamics": [
                "heat_transfer",
                "entropy",
                "energy_conservation",
                "phase_transitions",
            ],
            "electromagnetism": [
                "electric_fields",
                "magnetic_fields",
                "electromagnetic_waves",
                "circuits",
            ],
        },
        "sample_content": [
            "Mechanics studies the motion of objects and forces acting upon them.",
            "Thermodynamics deals with heat, work, and energy transformations.",
            "Electromagnetism describes electric and magnetic phenomena.",
            "Kinematics describes motion without considering its causes.",
            "Heat transfer involves the movement of thermal energy between systems.",
        ],
    },
    "biology": {
        "root_topics": ["genetics", "ecology", "evolution"],
        "subtopics": {
            "genetics": ["dna", "inheritance", "mutations", "gene_expression"],
            "ecology": ["ecosystems", "biodiversity", "food_chains", "conservation"],
            "evolution": ["natural_selection", "adaptation", "speciation", "phylogeny"],
        },
        "sample_content": [
            "Genetics studies heredity and variation in living organisms.",
            "Ecology examines relationships between organisms and their environment.",
            "Evolution explains how species change over time through natural processes.",
            "DNA carries genetic information in all living organisms.",
            "Ecosystems consist of interacting organisms and their environment.",
        ],
    },
}


class TreeOfTreesGenerator:
    """Generates a focused tree-of-trees structure with exactly 100 nodes."""

    def __init__(self):
        self.engine = None
        self.domain_nodes = {}  # domain -> list of node IDs
        self.topic_nodes = {}  # topic -> list of node IDs
        self.subtopic_nodes = {}  # subtopic -> list of node IDs
        self.all_nodes = []
        self.results = {
            "total_nodes": 0,
            "domains": {},
            "tree_structure": {},
            "connections": {"intra_domain": 0, "cross_domain": 0, "hierarchical": 0},
            "created_at": datetime.now().isoformat(),
        }

    def setup_engine(self, data_dir: str = "data/test_output"):
        """Set up the memory engine for tree generation."""
        logger.info("Setting up memory engine for tree-of-trees generation...")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Create memory engine with 100-node optimized settings
        self.engine = EnhancedMemoryEngineFactory.create_engine(
            data_path=f"{data_dir}/hp_mini.mex",
            ram_capacity=1000,
            flush_interval=5.0,
            flush_threshold=10,
        )

        logger.info("✅ Memory engine ready for tree generation")

    def generate_tree_of_trees(self):
        """Generate the complete tree-of-trees structure with 100 nodes."""
        logger.info("🌳 Generating Tree-of-Trees structure with 100 nodes...")

        start_time = time.time()

        # Phase 1: Create hierarchical structure within each domain
        for domain_name, domain_data in DOMAINS.items():
            logger.info(f"Creating {domain_name} tree...")
            self._create_domain_tree(domain_name, domain_data)

        # Phase 2: Create cross-domain connections
        logger.info("Creating cross-domain connections...")
        self._create_cross_domain_links()

        # Phase 3: Add additional intra-domain connections
        logger.info("Adding intra-domain connections...")
        self._add_intra_domain_links()

        # Calculate results
        end_time = time.time()
        self.results["total_nodes"] = len(self.all_nodes)
        self.results["generation_time"] = end_time - start_time

        logger.info(
            f"✅ Generated {len(self.all_nodes)} nodes in {end_time - start_time:.2f}s"
        )
        return self.results

    def _create_domain_tree(self, domain_name: str, domain_data: Dict[str, Any]):
        """Create a hierarchical tree structure for a single domain."""
        domain_node_ids = []

        # Level 1: Domain root node
        root_content = f"This is the root concept of {domain_name}. It encompasses all major topics and serves as the foundation for understanding this knowledge domain."
        root_node_id = self.engine.insert_memory(
            content=root_content,
            node_type="domain_root",
            tags=[domain_name, "root", "level_1", "tree_structure"],
            related_ids=[],
        )
        domain_node_ids.append(root_node_id)
        self.all_nodes.append(root_node_id)

        # Level 2: Root topics (3-4 per domain)
        topic_node_ids = []
        for topic in domain_data["root_topics"]:
            topic_content = (
                f"{topic.replace('_', ' ').title()} is a fundamental area within {domain_name}. "
                + random.choice(domain_data["sample_content"])
            )
            topic_node_id = self.engine.insert_memory(
                content=topic_content,
                node_type="topic",
                tags=[domain_name, topic, "level_2", "tree_structure"],
                related_ids=[root_node_id],  # Connect to domain root
            )
            topic_node_ids.append(topic_node_id)
            domain_node_ids.append(topic_node_id)
            self.all_nodes.append(topic_node_id)

            # Track by topic
            if topic not in self.topic_nodes:
                self.topic_nodes[topic] = []
            self.topic_nodes[topic].append(topic_node_id)

        # Level 3: Subtopics (connect to their parent topics)
        subtopic_node_ids = []
        for topic, subtopics in domain_data["subtopics"].items():
            # Find the parent topic node
            parent_topic_nodes = [nid for nid in topic_node_ids if topic in str(nid)]
            if not parent_topic_nodes:
                # Find by topic name in our tracking
                parent_topic_nodes = self.topic_nodes.get(topic, [])

            parent_node_id = (
                parent_topic_nodes[0] if parent_topic_nodes else root_node_id
            )

            for subtopic in subtopics:
                subtopic_content = (
                    f"{subtopic.replace('_', ' ').title()} is a specialized area within {topic.replace('_', ' ')}. "
                    + random.choice(domain_data["sample_content"])
                )
                subtopic_node_id = self.engine.insert_memory(
                    content=subtopic_content,
                    node_type="subtopic",
                    tags=[domain_name, topic, subtopic, "level_3", "tree_structure"],
                    related_ids=[parent_node_id],  # Connect to parent topic
                )
                subtopic_node_ids.append(subtopic_node_id)
                domain_node_ids.append(subtopic_node_id)
                self.all_nodes.append(subtopic_node_id)

                # Track by subtopic
                if subtopic not in self.subtopic_nodes:
                    self.subtopic_nodes[subtopic] = []
                self.subtopic_nodes[subtopic].append(subtopic_node_id)

        # Add remaining nodes to reach exactly 20 per domain
        remaining_nodes = NODES_PER_DOMAIN - len(domain_node_ids)
        for i in range(remaining_nodes):
            # Create concept nodes that connect to subtopics
            parent_subtopic = (
                random.choice(subtopic_node_ids) if subtopic_node_ids else root_node_id
            )
            concept_content = (
                f"Concept {i+1} in {domain_name}: "
                + random.choice(domain_data["sample_content"])
                + f" This represents a specific implementation or example within the domain."
            )

            concept_node_id = self.engine.insert_memory(
                content=concept_content,
                node_type="concept",
                tags=[domain_name, f"concept_{i+1}", "level_4", "tree_structure"],
                related_ids=[parent_subtopic],
            )
            domain_node_ids.append(concept_node_id)
            self.all_nodes.append(concept_node_id)

        # Store domain results
        self.domain_nodes[domain_name] = domain_node_ids
        self.results["domains"][domain_name] = {
            "node_count": len(domain_node_ids),
            "root_topics": len(domain_data["root_topics"]),
            "subtopics": sum(
                len(subtopics) for subtopics in domain_data["subtopics"].values()
            ),
            "tree_depth": 4,
        }

        logger.info(f"✅ Created {len(domain_node_ids)} nodes for {domain_name} tree")

    def _create_cross_domain_links(self):
        """Create connections between different domain trees."""
        cross_links_created = 0

        for _ in range(CROSS_REFERENCES):
            # Select two different domains
            domain_names = list(self.domain_nodes.keys())
            if len(domain_names) >= 2:
                domain1, domain2 = random.sample(domain_names, 2)

                # Select random nodes from each domain (prefer higher-level nodes)
                nodes1 = self.domain_nodes[domain1]
                nodes2 = self.domain_nodes[domain2]

                node1 = random.choice(
                    nodes1[:10]
                )  # Prefer first 10 nodes (higher level)
                node2 = random.choice(nodes2[:10])

                # Create bidirectional link by adding a new connection
                try:
                    # Add cross-domain reference
                    self.engine.link_memories(
                        node1, node2, link_type="cross_domain", weight=0.5
                    )
                    cross_links_created += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to create cross-domain link: {e} {traceback.print_exc()}"
                    )

        self.results["connections"]["cross_domain"] = cross_links_created
        logger.info(f"✅ Created {cross_links_created} cross-domain connections")

    def _add_intra_domain_links(self):
        """Add additional connections within each domain for richer structure."""
        intra_links_created = 0

        for domain_name, domain_nodes in self.domain_nodes.items():
            domain_links = 0

            # Create some additional connections within the domain
            for _ in range(INTRA_DOMAIN_LINKS // NUM_DOMAINS):
                if len(domain_nodes) >= 2:
                    node1, node2 = random.sample(domain_nodes, 2)

                    try:
                        self.engine.link_memories(
                            node1, node2, link_type="related", weight=0.7
                        )
                        domain_links += 1
                        intra_links_created += 1
                    except Exception as e:
                        logger.debug(f"Link already exists or failed: {e}")

            logger.info(f"Added {domain_links} intra-domain links for {domain_name}")

        self.results["connections"]["intra_domain"] = intra_links_created
        logger.info(f"✅ Created {intra_links_created} intra-domain connections")

    def save_results(self, data_dir: str = "data/test_output"):
        """Save the generation results and statistics."""
        results_file = f"{data_dir}/hp_mini_results.json"

        try:
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"✅ Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_summary(self):
        """Print a summary of the generated tree structure."""
        logger.info("=" * 60)
        logger.info("TREE-OF-TREES GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Nodes Generated: {self.results['total_nodes']}")
        logger.info(
            f"Generation Time: {self.results.get('generation_time', 0):.2f} seconds"
        )
        logger.info("")

        logger.info("Domain Breakdown:")
        for domain, stats in self.results["domains"].items():
            logger.info(
                f"  {domain}: {stats['node_count']} nodes (depth: {stats['tree_depth']})"
            )

        logger.info("")
        logger.info("Connection Statistics:")
        logger.info(
            f"  Cross-domain links: {self.results['connections']['cross_domain']}"
        )
        logger.info(
            f"  Intra-domain links: {self.results['connections']['intra_domain']}"
        )

        logger.info("=" * 60)


def main():
    """Main function to generate tree-of-trees test data."""
    logger.info("🌳 Starting Tree-of-Trees Test Data Generation (100 nodes)")

    try:
        # Create generator
        generator = TreeOfTreesGenerator()

        # Setup engine
        generator.setup_engine()

        # Generate tree structure
        results = generator.generate_tree_of_trees()

        # Save results
        generator.save_results()

        # Print summary
        generator.print_summary()

        logger.info("🎉 Tree-of-trees generation completed successfully!")

        # Launch database viewer
        launch_viewer = (
            input("\nLaunch database viewer to visualize the tree structure? (y/n): ")
            .lower()
            .strip()
        )
        if launch_viewer in ["y", "yes"]:
            logger.info("Launching database viewer...")
            os.system("python launcher.py --viewer")

        return True

    except Exception as e:
        logger.error(f"Tree generation failed: {e}")
        return False


if __name__ == "__main__":
    main()
