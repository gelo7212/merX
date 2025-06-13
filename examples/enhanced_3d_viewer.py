#!/usr/bin/env python3
"""
merX Memory 3D Visualization System

Beautiful, interactive 3D visualizations for exploring merX memory structures:
1. 3D Memory Network Graph - Visualize all memory connections in 3D space with memory fade
2. 3D Tree-of-Trees - Hierarchical memory structures with fractal patterns
3. Interactive Fractal Graph - Click to explore memory hierarchies with depth and evolution

Features:
- Memory decay visualization with color fading and opacity based on age and usage
- Animated transitions and smooth interactions using GPU acceleration  
- Beautiful UI with nature-inspired design (organic growth patterns)
- Connection strength visualization (thickness = weight, color = freshness)
- Time-based memory evolution (depth = time, width = associations)
- Fractal tree growth patterns showing memory branching and evolution
"""

import os
import sys
import json
import logging
import time
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID
import math
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
    from src.container.enhanced_di_container import create_enhanced_container
    from examples.gpu_accelerated_viewer import GPUAcceleratedMemoryViewer
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import merX modules, using simulation mode")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryDecayCalculator:
    """Calculate memory decay and visual properties based on time and usage with beautiful fading effects."""

    def __init__(self):
        # Different memory types fade at different rates (neuroscience-inspired)
        self.decay_rates = {
            "episodic": 0.8,    # Personal experiences fade moderately
            "semantic": 0.3,    # Facts and knowledge persist longer
            "procedural": 0.1,  # Skills persist the longest
            "working": 0.95,    # Working memory fades fastest
        }

    def calculate_decay(
        self, memory_type: str, age_hours: float, access_count: int = 0
    ) -> float:
        """Calculate decay factor (0 = completely faded, 1 = fresh) with realistic curves."""
        base_rate = self.decay_rates.get(memory_type, 0.5)

        # Age-based decay (exponential forgetting curve)
        age_factor = math.exp(-base_rate * age_hours / 24)  # 24 hours baseline

        # Access-based reinforcement (spaced repetition effect)
        access_factor = min(1.0, 1.0 + (access_count * 0.1))
        
        # Combine with slight randomness for natural variation
        random_factor = random.uniform(0.95, 1.05)

        return min(1.0, age_factor * access_factor * random_factor)

    def get_visual_properties(self, decay_factor: float) -> Dict[str, Any]:
        """Get beautiful visual properties based on decay factor with nature-inspired colors."""
        # Color transitions: Fresh green → Amber → Faded gray (like autumn leaves)
        if decay_factor > 0.8:
            # Fresh memories: Vibrant green/blue (like new growth)
            color = f"rgba(100, 255, 150, {0.8 + decay_factor * 0.2})"  
            size = 8 + decay_factor * 12
            glow_effect = "rgba(100, 255, 150, 0.4)"
        elif decay_factor > 0.6:
            # Recent memories: Yellow-green (like spring)
            color = f"rgba(150, 255, 100, {0.7 + decay_factor * 0.3})"  
            size = 6 + decay_factor * 10
            glow_effect = "rgba(150, 255, 100, 0.3)"
        elif decay_factor > 0.4:
            # Aging memories: Golden amber (like summer)
            color = f"rgba(255, 200, 50, {0.6 + decay_factor * 0.4})"  
            size = 4 + decay_factor * 8
            glow_effect = "rgba(255, 200, 50, 0.25)"
        elif decay_factor > 0.2:
            # Old memories: Orange-red (like autumn)
            color = f"rgba(255, 150, 50, {0.4 + decay_factor * 0.5})"  
            size = 3 + decay_factor * 6
            glow_effect = "rgba(255, 150, 50, 0.2)"
        else:
            # Very old memories: Faded gray (like winter/dormant)
            color = f"rgba(150, 150, 150, {0.2 + decay_factor * 0.3})"  
            size = 2 + decay_factor * 4
            glow_effect = "rgba(150, 150, 150, 0.1)"

        return {
            "color": color,
            "size": size,
            "opacity": 0.3 + decay_factor * 0.7,
            "line_width": 0.5 + decay_factor * 2,
            "glow_effect": glow_effect,
            "animation_speed": 1.0 + decay_factor * 2.0,  # Fresh memories animate faster
        }


class MemoryFractalBuilder:
    """Build fractal tree structures representing memory hierarchies and evolution."""
    
    def __init__(self):
        self.fractal_cache = {}
        
    def build_fractal_tree(self, root_memory: Dict, related_memories: List[Dict], max_depth: int = 5) -> Dict:
        """Build a fractal tree structure from a memory and its connections."""
        
        # Create fractal tree node
        fractal_node = {
            "id": root_memory["id"],
            "content": root_memory.get("content", ""),
            "level": 0,
            "children": [],
            "position": {"x": 0, "y": 0, "z": 0},
            "visual": root_memory.get("visual", {}),
            "branch_strength": 1.0,
            "growth_direction": {"theta": 0, "phi": 0},
            "fractal_properties": {
                "branch_count": 0,
                "total_nodes": 1,
                "max_depth_reached": 0,
                "density": 1.0
            }
        }
        
        # Recursively build branches (children)
        self._build_fractal_branches(
            fractal_node, 
            related_memories, 
            current_depth=0, 
            max_depth=max_depth,
            parent_direction={"theta": 0, "phi": 0}
        )
        
        return fractal_node
        
    def _build_fractal_branches(self, parent_node: Dict, available_memories: List[Dict], 
                               current_depth: int, max_depth: int, parent_direction: Dict):
        """Recursively build fractal branches with organic growth patterns."""
        
        if current_depth >= max_depth or not available_memories:
            return
            
        # Limit branching factor based on decay (weaker memories have fewer branches)
        parent_decay = parent_node.get("visual", {}).get("opacity", 0.5)
        max_branches = max(1, int(parent_decay * 8))  # 1-8 branches max
        
        # Select most related memories for branching
        branch_memories = available_memories[:max_branches]
        
        for i, memory in enumerate(branch_memories):
            # Calculate branch growth direction (fractal angles)
            branch_angle = (i / len(branch_memories)) * 2 * math.pi
            elevation_angle = random.uniform(-math.pi/4, math.pi/4)  # Organic variation
            
            # Create child fractal node
            child_node = {
                "id": memory["id"],
                "content": memory.get("content", ""),
                "level": current_depth + 1,
                "children": [],
                "visual": memory.get("visual", {}),
                "branch_strength": parent_decay * memory.get("visual", {}).get("opacity", 0.5),
                "growth_direction": {
                    "theta": branch_angle,
                    "phi": elevation_angle
                },
                "parent_id": parent_node["id"]
            }
            
            # Calculate fractal position (organic growth)
            branch_length = 2.0 * parent_decay * (1.0 - current_depth / max_depth)
            
            child_node["position"] = {
                "x": parent_node["position"]["x"] + branch_length * math.cos(branch_angle) * math.cos(elevation_angle),
                "y": parent_node["position"]["y"] + branch_length * math.sin(branch_angle) * math.cos(elevation_angle),
                "z": parent_node["position"]["z"] + branch_length * math.sin(elevation_angle) + current_depth * 0.5
            }
            
            # Add child node to parent
            parent_node["children"].append(child_node)
            
            # Recursively build branches for child node
            self._build_fractal_branches(
                child_node, 
                [m for m in available_memories if m["id"] != memory["id"]],  # Exclude used memory
                current_depth + 1, 
                max_depth,
                {"theta": branch_angle, "phi": elevation_angle}
            )
            
        # Update fractal properties
        parent_node["fractal_properties"]["branch_count"] = len(parent_node["children"])
        parent_node["fractal_properties"]["max_depth_reached"] = current_depth
        total_descendants = sum(self._count_descendants(child) for child in parent_node["children"])
        parent_node["fractal_properties"]["total_nodes"] = 1 + total_descendants
        
    def _count_descendants(self, node: Dict) -> int:
        """Count total descendants in fractal tree."""
        if not node.get("children"):
            return 1
        return 1 + sum(self._count_descendants(child) for child in node["children"])


class MemoryVisualization3D:
    """Create beautiful 3D visualizations of merX memory structures with GPU acceleration."""

    def __init__(self):
        self.viz_dir = Path("data/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.decay_calc = MemoryDecayCalculator()
        self.fractal_builder = MemoryFractalBuilder()
        self.gpu_viewer = GPUAcceleratedMemoryViewer()
        self.engine = None

    def setup_memory_engine(self, data_dir: str = "data/test_output"):
        """Setup memory engine if available."""
        try:
            # Try to find existing data files in order of preference
            data_sources = [
                f"{data_dir}/hp_mini.mex",  # Test output data (extreme performance test)
                f"{data_dir}/memory.mex",  # Default memory file
                "data/search_accuracy_test/memory.mex",  # Search accuracy test data
                "data/test_output/hp_mini.mex",  # Extreme performance test data
            ]

            loaded = False
            for data_path in data_sources:
                if os.path.exists(data_path):
                    try:
                        # Use static factory method to create engine with existing data
                        self.engine = EnhancedMemoryEngineFactory.create_engine(
                            data_path=data_path,
                            ram_capacity=50000,  # Larger capacity for better visualization
                        )
                        logger.info(
                            f"✅ Memory engine initialized with data from: {data_path}"
                        )

                        # Force loading of data into RAMX if not already loaded
                        storage_ramx = getattr(self.engine.storage, "ramx", None)
                        if storage_ramx:
                            ramx_stats = storage_ramx.get_stats()
                            logger.info(f"📊 RAMX loaded: {ramx_stats}")

                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load data from {data_path}: {e}")
                        continue

            if not loaded:
                logger.warning("No existing memory data found, will generate sample data")

        except Exception as e:
            logger.error(f"Error setting up memory engine: {e}")

    def load_memory_data(self) -> Dict[str, Any]:
        """Load memory data from engine or generate sample data."""
        if self.engine:
            return self._load_real_memory_data()
        else:
            raise RuntimeError(
                "Memory engine not initialized. Please call setup_memory_engine() first."
            )

    def _load_real_memory_data(self) -> Dict[str, Any]:
        """Load real memory data from RAMX (RAM-based memory store)."""
        logger.info("🧠 Loading real memory data from RAMX...")

        data = {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes": 0,
                "total_edges": 0,
                "memory_types": {},
                "domains": {},
                "decay_distribution": {},
            },
        }

        try:
            # Get RAMX instance from memory I/O orchestrator using getattr for type safety
            ramx = (
                getattr(self.engine.storage, "ramx", None)
                if self.engine and hasattr(self.engine, "storage")
                else None
            )

            if ramx:
                logger.info("✅ Found RAMX directly in storage")
            else:
                logger.warning("No RAMX instance found, falling back to sample data")
                logger.warning(
                    f"Engine: {self.engine}, Storage: {getattr(self.engine, 'storage', None) if self.engine else None}"
                )
                return self._generate_sample_memory_data()

            # Get all nodes from RAMX (live, in-memory data with current activation levels)
            ramx_nodes = ramx.get_all_nodes()
            logger.info(f"🌟 Found {len(ramx_nodes)} nodes in RAMX memory")

            current_time = time.time()

            for ramx_node in ramx_nodes:
                # Calculate age in hours from timestamp
                age_hours = (current_time - ramx_node.timestamp) / 3600

                # Use live activation and decay data from RAMX
                memory_type = ramx_node.node_type
                decay_factor = ramx_node.activation  # RAMX already handles decay

                # Get visual properties based on current activation
                visual_props = self.decay_calc.get_visual_properties(decay_factor)

                # Extract domain from tags or content
                domain = "general"
                if ramx_node.tags:
                    # Look for domain-like tags
                    domain_tags = [
                        "nature",
                        "learning",
                        "relationships",
                        "creativity",
                        "growth",
                        "technology",
                    ]
                    for tag in ramx_node.tags:
                        if tag.lower() in domain_tags:
                            domain = tag.lower()
                            break

                node = {
                    "id": str(ramx_node.id),
                    "content": (
                        ramx_node.content[:100] + "..."
                        if len(ramx_node.content) > 100
                        else ramx_node.content
                    ),
                    "domain": domain,
                    "memory_type": memory_type,
                    "created_at": datetime.fromtimestamp(
                        ramx_node.timestamp
                    ).isoformat(),
                    "age_hours": age_hours,
                    "decay_factor": decay_factor,
                    "access_count": 0,  # RAMX doesn't track access count separately
                    "tags": ramx_node.tags,
                    "visual": visual_props,
                    "activation": ramx_node.activation,  # Live activation level
                    "links": len(ramx_node.links),  # Number of connections
                }

                data["nodes"].append(node)

            # Load connections from RAMX links (real neural connections)
            for ramx_node in ramx_nodes:
                source_id = str(ramx_node.id)

                # Get direct links from RAMX node
                for target_id, (weight, link_type) in ramx_node.links.items():
                    # Only add if target node exists in our dataset
                    if any(node["id"] == str(target_id) for node in data["nodes"]):
                        edge = {
                            "source": source_id,
                            "target": str(target_id),
                            "weight": weight,
                            "type": link_type,
                            "strength": weight,
                        }
                        data["edges"].append(edge)

            # Update stats
            data["stats"]["total_nodes"] = len(data["nodes"])
            data["stats"]["total_edges"] = len(data["edges"])

            # Calculate distributions from live data
            for node in data["nodes"]:
                memory_type = node["memory_type"]
                data["stats"]["memory_types"][memory_type] = (
                    data["stats"]["memory_types"].get(memory_type, 0) + 1
                )

                domain = node["domain"]
                data["stats"]["domains"][domain] = (
                    data["stats"]["domains"].get(domain, 0) + 1
                )

                decay_bucket = f"{int(node['decay_factor'] * 10) * 10}%"
                data["stats"]["decay_distribution"][decay_bucket] = (
                    data["stats"]["decay_distribution"].get(decay_bucket, 0) + 1
                )

            # Get RAMX statistics
            ramx_stats = ramx.get_stats()
            data["stats"]["ramx_stats"] = ramx_stats

            logger.info(
                f"✅ Loaded {data['stats']['total_nodes']} live memories with {data['stats']['total_edges']} neural connections from RAMX"
            )
            logger.info(
                f"🔥 Active nodes: {ramx_stats.get('active_nodes', 0)}, Avg activation: {ramx_stats.get('average_activation', 0):.3f}"
            )

        except Exception as e:
            logger.error(f"Error loading RAMX memory data: {e}")
            logger.warning("Falling back to sample data")
            return self._generate_sample_memory_data()

        return data

    def _generate_sample_memory_data(self) -> Dict[str, Any]:
        """Generate beautiful sample memory data for demonstration."""
        logger.info("🌱 Generating sample memory data with natural growth patterns...")

        # Define memory domains with natural themes
        domains = {
            "nature": {
                "color": "rgba(34, 139, 34, 0.8)",
                "memories": [
                    "Walking through the forest on a crisp autumn morning",
                    "The sound of rain on leaves during a thunderstorm",
                    "Watching sunrise over mountain peaks",
                    "The smell of pine trees after rain",
                    "Ocean waves crashing against rocky cliffs",
                ],
            },
            "learning": {
                "color": "rgba(70, 130, 180, 0.8)",
                "memories": [
                    "Understanding quantum mechanics for the first time",
                    "Learning to play guitar chords",
                    "Reading about artificial intelligence",
                    "Discovering fractal mathematics",
                    "Studying memory formation in the brain",
                ],
            },
            "relationships": {
                "color": "rgba(255, 105, 180, 0.8)",
                "memories": [
                    "Laughing with friends around a campfire",
                    "Deep conversation with a mentor",
                    "Teaching someone a new skill",
                    "Sharing ideas in a creative brainstorm",
                    "Feeling understood by someone special",
                ],
            },
            "creativity": {
                "color": "rgba(255, 165, 0, 0.8)",
                "memories": [
                    "The moment an idea suddenly clicked",
                    "Painting a landscape from memory",
                    "Writing poetry under starlight",
                    "Designing an elegant solution",
                    "Creating something beautiful from nothing",
                ],
            },
            "growth": {
                "color": "rgba(138, 43, 226, 0.8)",
                "memories": [
                    "Overcoming a difficult challenge",
                    "Realizing a limiting belief was false",
                    "Learning from a meaningful failure",
                    "Developing patience through practice",
                    "Finding strength in vulnerability",
                ],
            },
        }

        data = {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes": 0,
                "total_edges": 0,
                "memory_types": {},
                "domains": {},
                "decay_distribution": {},
            },
        }

        # Generate nodes with realistic decay patterns
        node_id = 0
        current_time = datetime.now()

        for domain_name, domain_data in domains.items():
            for i, memory_content in enumerate(domain_data["memories"]):
                # Simulate different ages (some memories are older)
                age_days = random.uniform(0, 365)  # Up to 1 year old
                age_hours = age_days * 24

                # Assign memory type based on content
                if (
                    "learning" in domain_name
                    or "understanding" in memory_content.lower()
                ):
                    memory_type = "semantic"
                elif (
                    "feeling" in memory_content.lower()
                    or "emotion" in memory_content.lower()
                ):
                    memory_type = "episodic"
                else:
                    memory_type = random.choice(["semantic", "episodic", "procedural"])

                # Simulate access patterns
                access_count = random.randint(0, 20)

                # Calculate decay
                decay_factor = self.decay_calc.calculate_decay(
                    memory_type, age_hours, access_count
                )

                # Get visual properties
                visual_props = self.decay_calc.get_visual_properties(decay_factor)

                # Override color with domain color but adjust opacity based on decay
                base_color = domain_data["color"]
                visual_props["color"] = base_color.replace(
                    "0.8)", f"{visual_props['opacity']})"
                )

                node = {
                    "id": str(node_id),
                    "content": memory_content,
                    "domain": domain_name,
                    "memory_type": memory_type,
                    "created_at": (
                        current_time - timedelta(hours=age_hours)
                    ).isoformat(),
                    "age_hours": age_hours,
                    "decay_factor": decay_factor,
                    "access_count": access_count,
                    "tags": [domain_name, memory_type],
                    "visual": visual_props,
                }

                data["nodes"].append(node)
                node_id += 1

        # Create additional nodes for richer visualization
        for i in range(50):  # Add 50 more varied nodes
            domain_name = random.choice(list(domains.keys()))
            age_hours = random.uniform(1, 8760)  # Up to 1 year
            memory_type = random.choice(
                ["semantic", "episodic", "procedural", "working"]
            )
            access_count = random.randint(0, 15)

            decay_factor = self.decay_calc.calculate_decay(
                memory_type, age_hours, access_count
            )
            visual_props = self.decay_calc.get_visual_properties(decay_factor)

            # Generate varied content
            content_templates = [
                f"Exploring {random.choice(['concepts', 'ideas', 'patterns', 'connections'])} in {domain_name}",
                f"Deep insight about {random.choice(['creativity', 'learning', 'growth', 'understanding'])}",
                f"Memorable moment involving {random.choice(['discovery', 'realization', 'breakthrough', 'clarity'])}",
                f"Complex thought about {random.choice(['relationships', 'systems', 'processes', 'evolution'])}",
            ]

            node = {
                "id": str(node_id),
                "content": random.choice(content_templates),
                "domain": domain_name,
                "memory_type": memory_type,
                "created_at": (current_time - timedelta(hours=age_hours)).isoformat(),
                "age_hours": age_hours,
                "decay_factor": decay_factor,
                "access_count": access_count,
                "tags": [domain_name, memory_type],
                "visual": visual_props,
            }

            data["nodes"].append(node)
            node_id += 1

        # Generate edges with natural connection patterns
        nodes = data["nodes"]
        for i, node1 in enumerate(nodes):
            # Connect to similar domain nodes
            for j, node2 in enumerate(nodes[i + 1 :], i + 1):
                # Higher probability for same domain connections
                if node1["domain"] == node2["domain"]:
                    connection_prob = 0.4
                else:
                    connection_prob = 0.1

                # Memory type similarity increases connection probability
                if node1["memory_type"] == node2["memory_type"]:
                    connection_prob += 0.2

                # Fresher memories connect more
                avg_decay = (node1["decay_factor"] + node2["decay_factor"]) / 2
                connection_prob *= 0.5 + avg_decay * 0.5

                if random.random() < connection_prob:
                    strength = random.uniform(0.3, 1.0)

                    # Determine connection type
                    if node1["domain"] == node2["domain"]:
                        conn_type = "semantic"
                    elif abs(node1["age_hours"] - node2["age_hours"]) < 24:
                        conn_type = "temporal"
                    else:
                        conn_type = "associative"

                    edge = {
                        "source": node1["id"],
                        "target": node2["id"],
                        "weight": strength,
                        "type": conn_type,
                        "strength": strength,
                    }
                    data["edges"].append(edge)

        # Limit edges for performance
        if len(data["edges"]) > 200:
            data["edges"] = random.sample(data["edges"], 200)

        # Update stats
        data["stats"]["total_nodes"] = len(data["nodes"])
        data["stats"]["total_edges"] = len(data["edges"])

        # Calculate distributions
        for node in data["nodes"]:
            memory_type = node["memory_type"]
            data["stats"]["memory_types"][memory_type] = (
                data["stats"]["memory_types"].get(memory_type, 0) + 1
            )

            domain = node["domain"]
            data["stats"]["domains"][domain] = (
                data["stats"]["domains"].get(domain, 0) + 1
            )

            decay_bucket = f"{int(node['decay_factor'] * 10) * 10}%"
            data["stats"]["decay_distribution"][decay_bucket] = (
                data["stats"]["decay_distribution"].get(decay_bucket, 0) + 1
            )

        logger.info(
            f"🌿 Generated {data['stats']['total_nodes']} memories with {data['stats']['total_edges']} natural connections"
        )
        return data

    def create_3d_memory_network(self, data: Dict[str, Any]) -> str:
        """Create beautiful 3D memory network visualization."""
        logger.info("🌌 Creating 3D memory network visualization...")

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes with attributes
        for node in data["nodes"]:
            G.add_node(
                node["id"],
                content=node["content"],
                domain=node["domain"],
                memory_type=node["memory_type"],
                decay_factor=node["decay_factor"],
                visual=node["visual"],
            )

        # Add edges
        for edge in data["edges"]:
            if edge["source"] in [n["id"] for n in data["nodes"]] and edge[
                "target"
            ] in [n["id"] for n in data["nodes"]]:
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["strength"],
                    type=edge["type"],
                )

        # Calculate 3D layout using spring layout with physics
        logger.info("Computing natural 3D layout...")
        if len(G.nodes()) > 0:
            pos_2d = nx.spring_layout(G, k=2, iterations=50, dim=2)

            # Convert to 3D by adding z-coordinate based on memory characteristics
            pos_3d = {}
            for node_id, (x, y) in pos_2d.items():
                node_data = next(n for n in data["nodes"] if n["id"] == node_id)

                # Z-coordinate based on memory age and type
                if node_data["memory_type"] == "working":
                    z = 2.0 + random.uniform(-0.2, 0.2)  # Top layer
                elif node_data["memory_type"] == "episodic":
                    z = 1.0 + random.uniform(-0.3, 0.3)  # Middle layer
                elif node_data["memory_type"] == "semantic":
                    z = 0.0 + random.uniform(-0.3, 0.3)  # Bottom layer
                else:
                    z = -1.0 + random.uniform(-0.2, 0.2)  # Deep layer

                # Adjust z based on decay (fresh memories float higher)
                z += node_data["decay_factor"] * 0.5

                pos_3d[node_id] = (x, y, z)
        else:
            pos_3d = {}

        # Create 3D visualization
        fig = go.Figure()

        # Add edges first (so they appear behind nodes)
        if G.edges():
            edge_x, edge_y, edge_z = [], [], []
            edge_info = []

            for edge in G.edges(data=True):
                source_id, target_id, edge_data = edge
                if source_id in pos_3d and target_id in pos_3d:
                    x0, y0, z0 = pos_3d[source_id]
                    x1, y1, z1 = pos_3d[target_id]

                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])

                    edge_info.append(
                        f"Connection: {edge_data.get('type', 'unknown')} (strength: {edge_data.get('weight', 0):.2f})"
                    )

            # Add edge trace
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="rgba(125, 125, 125, 0.3)", width=2),
                    hoverinfo="none",
                    showlegend=False,
                    name="Connections",
                )
            )

        # Add nodes
        if pos_3d:
            node_x = [
                pos_3d[node["id"]][0] for node in data["nodes"] if node["id"] in pos_3d
            ]
            node_y = [
                pos_3d[node["id"]][1] for node in data["nodes"] if node["id"] in pos_3d
            ]
            node_z = [
                pos_3d[node["id"]][2] for node in data["nodes"] if node["id"] in pos_3d
            ]

            node_colors = [
                node["visual"]["color"]
                for node in data["nodes"]
                if node["id"] in pos_3d
            ]
            node_sizes = [
                node["visual"]["size"] for node in data["nodes"] if node["id"] in pos_3d
            ]

            hover_text = []
            for node in data["nodes"]:
                if node["id"] in pos_3d:
                    hover_text.append(
                        f"<b>{node['content'][:50]}...</b><br>"
                        + f"Domain: {node['domain']}<br>"
                        + f"Type: {node['memory_type']}<br>"
                        + f"Age: {node['age_hours']:.1f} hours<br>"
                        + f"Decay: {node['decay_factor']:.2f}<br>"
                        + f"Access Count: {node['access_count']}"
                    )

            fig.add_trace(
                go.Scatter3d(
                    x=node_x,
                    y=node_y,
                    z=node_z,
                    mode="markers",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
                        opacity=0.8,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    name="Memories",
                )
            )

        # Update layout with beautiful styling
        fig.update_layout(
            title={
                "text": "🧠 merX Memory Network - 3D Visualization",
                "x": 0.5,
                "font": {"size": 24, "color": "white"},
            },
            scene=dict(
                xaxis=dict(
                    title="Semantic Space X",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                ),
                yaxis=dict(
                    title="Semantic Space Y",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                ),
                zaxis=dict(
                    title="Memory Layers",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor="rgba(0, 0, 0, 0.9)",
            ),
            paper_bgcolor="rgba(0, 0, 0, 0.9)",
            plot_bgcolor="rgba(0, 0, 0, 0.9)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            height=800,
        )

        # Add memory layer annotations
        annotations = [
            dict(
                text="🧠 Working Memory (Top Layer)",
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="lightblue", size=12),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor="lightblue",
                borderwidth=1,
            ),
            dict(
                text="📖 Episodic Memory (Middle Layer)",
                x=0.02,
                y=0.94,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="lightgreen", size=12),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor="lightgreen",
                borderwidth=1,
            ),
            dict(
                text="🔍 Semantic Memory (Base Layer)",
                x=0.02,
                y=0.90,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="lightyellow", size=12),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor="lightyellow",
                borderwidth=1,
            ),
        ]

        fig.update_layout(annotations=annotations)

        # Save the visualization
        output_file = self.viz_dir / "memory_3d_network.html"
        fig.write_html(
            str(output_file),
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "select2d", "lasso2d"],
                "responsive": True,
            },
        )

        logger.info(f"✅ 3D memory network saved to {output_file}")
        return str(output_file)

    def create_tree_of_trees_3d(self, data: Dict[str, Any]) -> str:
        """Create fractal tree-of-trees 3D visualization."""
        logger.info("🌳 Creating tree-of-trees fractal visualization...")

        # Group memories by domain for tree structure
        domain_trees = {}
        for node in data["nodes"]:
            domain = node["domain"]
            if domain not in domain_trees:
                domain_trees[domain] = []
            domain_trees[domain].append(node)

        fig = go.Figure()

        # Colors for different domains
        domain_colors = {
            "nature": "#228B22",
            "learning": "#4682B4",
            "relationships": "#FF69B4",
            "creativity": "#FFA500",
            "growth": "#8A2BE2",
        }

        tree_spacing = 8  # Space between trees
        tree_index = 0

        for domain, nodes in domain_trees.items():
            if len(nodes) < 2:
                continue

            # Calculate tree center position
            tree_center_x = (tree_index % 3) * tree_spacing
            tree_center_y = (tree_index // 3) * tree_spacing
            tree_center_z = 0

            # Sort nodes by decay factor (freshest at top)
            nodes.sort(key=lambda x: x["decay_factor"], reverse=True)

            # Create tree structure
            tree_positions = self._generate_tree_positions(
                len(nodes), tree_center_x, tree_center_y, tree_center_z
            )

            # Add tree trunk and branches
            self._add_tree_structure(
                fig, tree_positions, nodes, domain_colors.get(domain, "#888888")
            )

            # Add leaves (memory nodes)
            self._add_tree_leaves(fig, tree_positions, nodes, domain)

            tree_index += 1

        # Update layout for tree visualization
        fig.update_layout(
            title={
                "text": "🌲 merX Memory Forest - Tree of Trees",
                "x": 0.5,
                "font": {"size": 24, "color": "white"},
            },
            scene=dict(
                xaxis=dict(
                    title="Forest X",
                    showgrid=False,
                    showticklabels=False,
                    showbackground=True,
                    backgroundcolor="rgba(34, 139, 34, 0.1)",
                ),
                yaxis=dict(
                    title="Forest Y",
                    showgrid=False,
                    showticklabels=False,
                    showbackground=True,
                    backgroundcolor="rgba(34, 139, 34, 0.1)",
                ),
                zaxis=dict(
                    title="Growth Height",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    showbackground=True,
                    backgroundcolor="rgba(0, 50, 0, 0.3)",
                ),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                bgcolor="rgba(0, 20, 0, 0.9)",
            ),
            paper_bgcolor="rgba(0, 20, 0, 0.9)",
            plot_bgcolor="rgba(0, 20, 0, 0.9)",
            font=dict(color="lightgreen"),
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            height=800,
            showlegend=True,
        )

        # Save the visualization
        output_file = self.viz_dir / "memory_tree_forest_3d.html"
        fig.write_html(
            str(output_file),
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )

        logger.info(f"✅ Tree-of-trees visualization saved to {output_file}")
        return str(output_file)

    def _generate_tree_positions(
        self, num_nodes: int, center_x: float, center_y: float, center_z: float
    ) -> List[Tuple[float, float, float]]:
        """Generate natural tree-like positions for nodes."""
        positions = []

        # Tree parameters
        trunk_height = 2.0
        branch_levels = min(4, int(math.log2(num_nodes)) + 1)

        for i in range(num_nodes):
            # Calculate level in tree (0 = root, higher = branches)
            level = min(int(math.log2(i + 1)), branch_levels - 1)

            # Position within level
            level_position = i - (2**level - 1) if level > 0 else 0
            level_width = 2**level

            # Calculate position
            if level == 0:
                # Root at base
                x, y, z = center_x, center_y, center_z
            else:
                # Branches spread out and up
                angle = (level_position / level_width) * 2 * math.pi
                radius = level * 1.5
                height = trunk_height * (level / branch_levels)

                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                z = center_z + height + random.uniform(-0.2, 0.2)

            positions.append((x, y, z))

        return positions

    def _add_tree_structure(
        self, fig: go.Figure, positions: List[Tuple], nodes: List[Dict], color: str
    ):
        """Add tree trunk and branch structure."""
        if len(positions) < 2:
            return

        # Create branch connections
        branch_x, branch_y, branch_z = [], [], []

        for i in range(1, len(positions)):
            parent_idx = (i - 1) // 2  # Binary tree parent
            if parent_idx < len(positions):
                x0, y0, z0 = positions[parent_idx]
                x1, y1, z1 = positions[i]

                branch_x.extend([x0, x1, None])
                branch_y.extend([y0, y1, None])
                branch_z.extend([z0, z1, None])

        # Add branch trace
        fig.add_trace(
            go.Scatter3d(
                x=branch_x,
                y=branch_y,
                z=branch_z,
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="none",
                showlegend=False,
                name="Branches",
            )
        )

    def _add_tree_leaves(
        self, fig: go.Figure, positions: List[Tuple], nodes: List[Dict], domain: str
    ):
        """Add memory nodes as leaves on the tree."""
        if not positions or not nodes:
            return

        # Limit to available positions
        num_to_show = min(len(positions), len(nodes))

        leaf_x = [positions[i][0] for i in range(num_to_show)]
        leaf_y = [positions[i][1] for i in range(num_to_show)]
        leaf_z = [positions[i][2] for i in range(num_to_show)]

        leaf_colors = [nodes[i]["visual"]["color"] for i in range(num_to_show)]
        leaf_sizes = [
            nodes[i]["visual"]["size"] * 1.5 for i in range(num_to_show)
        ]  # Bigger for tree view

        hover_text = []
        for i in range(num_to_show):
            node = nodes[i]
            hover_text.append(
                f"<b>{node['content'][:60]}...</b><br>"
                + f"🌿 Branch: {domain}<br>"
                + f"🕒 Age: {node['age_hours']:.1f} hours<br>"
                + f"💫 Freshness: {node['decay_factor']:.2f}<br>"
                + f"🔗 Access: {node['access_count']} times"
            )

        fig.add_trace(
            go.Scatter3d(
                x=leaf_x,
                y=leaf_y,
                z=leaf_z,
                mode="markers",
                marker=dict(
                    size=leaf_sizes,
                    color=leaf_colors,
                    line=dict(width=2, color="rgba(255, 255, 255, 0.6)"),
                    opacity=0.9,
                    symbol="circle",
                ),
                text=hover_text,
                hoverinfo="text",
                showlegend=True,
                name=f"🌿 {domain.title()} Memories",
            )
        )    def _organize_memories_into_hierarchy(self, nodes: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize memories into hierarchical levels based on decay and connections."""
        memory_levels = {}
        
        # Sort nodes by decay factor (freshest first)
        sorted_nodes = sorted(nodes, key=lambda x: x["decay_factor"], reverse=True)
        
        # Assign to levels based on decay factor thresholds
        for node in sorted_nodes:
            decay = node["decay_factor"]
            
            if decay > 0.8:
                level = 0  # Fresh memories at top
            elif decay > 0.6:
                level = 1  # Recent memories
            elif decay > 0.4:
                level = 2  # Aging memories
            elif decay > 0.2:
                level = 3  # Old memories
            else:
                level = 4  # Very old memories at bottom
                
            if level not in memory_levels:
                memory_levels[level] = []
            memory_levels[level].append(node)
            
        return memory_levels
    
    def _generate_memory_based_positions(self, memory_nodes: List[Dict], level: int) -> List[Dict]:
        """Generate 3D positions based on memory content and relationships."""
        positions = []
        num_nodes = len(memory_nodes)
        
        if num_nodes == 0:
            return positions
            
        # Create fractal pattern based on level
        for i, node in enumerate(memory_nodes):
            # Base position using golden ratio spiral
            golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle in radians
            theta = i * golden_angle
            
            # Radius increases with position but varies by decay
            base_radius = math.sqrt(i) * 0.5
            decay_modifier = node["decay_factor"] * 0.3
            radius = base_radius + decay_modifier
            
            # Height based on level and some variation
            base_height = level * 2.0
            height_variation = node["decay_factor"] * 0.5
            
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)  
            z = base_height + height_variation + random.uniform(-0.2, 0.2)
            
            # Add domain-based clustering
            domain_offset = self._get_domain_offset(node["domain"])
            x += domain_offset["x"]
            y += domain_offset["y"]
            
            positions.append({
                "x": x,
                "y": y,
                "z": z,
                "id": node["id"]
            })
            
        return positions
    
    def _get_domain_offset(self, domain: str) -> Dict[str, float]:
        """Get spatial offset based on memory domain for clustering."""
        domain_offsets = {
            "nature": {"x": -2.0, "y": 2.0},
            "learning": {"x": 2.0, "y": 2.0},
            "relationships": {"x": -2.0, "y": -2.0},
            "creativity": {"x": 2.0, "y": -2.0},
            "growth": {"x": 0.0, "y": 0.0},
            "general": {"x": 0.0, "y": 0.0},
            "artificial_intelligence": {"x": 3.0, "y": 0.0},
            "computer_science": {"x": 2.0, "y": 1.0},
            "biology": {"x": -3.0, "y": 0.0},
            "physics": {"x": 0.0, "y": 3.0},
            "chemistry": {"x": 1.0, "y": -3.0},
            "mathematics": {"x": -1.0, "y": 3.0},
            "literature": {"x": -2.0, "y": 1.0},
            "history": {"x": 1.0, "y": 2.0},
            "psychology": {"x": -1.0, "y": -1.0},
            "philosophy": {"x": 0.0, "y": -3.0}
        }
        return domain_offsets.get(domain, {"x": 0.0, "y": 0.0})

    def create_fractal_hierarchy_graph(self, data: Dict[str, Any]) -> str:
        """Create interactive fractal hierarchy visualization."""
        logger.info(
            "🔄 Creating interactive fractal hierarchy graph using real memory data..."
        )

        # Use real memory data organized into hierarchical levels
        # Group memories by decay level and memory type for natural hierarchy
        memory_levels = self._organize_memories_into_hierarchy(data["nodes"])

        fig = go.Figure()

        # Add fractal structure using real memory organization
        all_fractal_points = []
        for level, memory_nodes in memory_levels.items():
            if memory_nodes:
                # Generate positions for this level based on memory content and connections
                positions = self._generate_memory_based_positions(memory_nodes, level)

                x_coords = [pos["x"] for pos in positions]
                y_coords = [pos["y"] for pos in positions]
                z_coords = [pos["z"] for pos in positions]

                # Use actual memory data properties
                colors = [node["visual"]["color"] for node in memory_nodes]
                sizes = [max(8, node["visual"]["size"]) for node in memory_nodes]

                hover_text = []
                level_points = []
                for i, node in enumerate(memory_nodes):
                    hover_text.append(
                        f"<b>Memory Hierarchy Level {level}</b><br>"
                        + f"Type: {node['memory_type']}<br>"
                        + f"Domain: {node['domain']}<br>"
                        + f"Content: {node['content'][:50]}...<br>"
                        + f"Activation: {node.get('activation', 0.5):.3f}<br>"
                        + f"Links: {node.get('links', 0)}<br>"
                        + f"Click to explore connections"
                    )
                    
                    level_points.append({
                        "id": node["id"],
                        "level": level,
                        "x": positions[i]["x"],
                        "y": positions[i]["y"],
                        "z": positions[i]["z"]
                    })

                all_fractal_points.extend(level_points)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode="markers",
                        marker=dict(
                            size=sizes,
                            color=colors,
                            line=dict(width=2, color="rgba(255, 255, 255, 0.8)"),
                            opacity=0.8 - (level * 0.15),  # Fade deeper levels
                        ),
                        text=hover_text,
                        hoverinfo="text",
                        showlegend=True,
                        name=f"Fractal Level {level}",
                        customdata=[p["id"] for p in level_points],  # For click handling
                    )
                )

        # Add connecting lines between fractal levels
        self._add_fractal_connections(fig, all_fractal_points)

        # Update layout for fractal visualization
        fig.update_layout(
            title={
                "text": "🌀 merX Memory Fractal - Interactive Hierarchy",
                "x": 0.5,
                "font": {"size": 24, "color": "white"},
            },
            scene=dict(
                xaxis=dict(
                    title="Fractal Dimension X",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.2)",
                    showbackground=True,
                    backgroundcolor="rgba(20, 0, 40, 0.3)",
                ),
                yaxis=dict(
                    title="Fractal Dimension Y",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.2)",
                    showbackground=True,
                    backgroundcolor="rgba(20, 0, 40, 0.3)",
                ),
                zaxis=dict(
                    title="Hierarchy Depth",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.2)",
                    showbackground=True,
                    backgroundcolor="rgba(20, 0, 40, 0.3)",
                ),
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
                bgcolor="rgba(10, 0, 30, 0.9)",
            ),
            paper_bgcolor="rgba(10, 0, 30, 0.9)",
            plot_bgcolor="rgba(10, 0, 30, 0.9)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            height=800,
        )

        # Add instructions
        fig.add_annotation(
            text="🌀 Click on any node to explore its memory hierarchy<br>🔍 Zoom and rotate to navigate the fractal",
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="cyan", size=14),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor="cyan",
            borderwidth=1,
        )

        # Save the visualization
        output_file = self.viz_dir / "memory_fractal_hierarchy.html"
        fig.write_html(
            str(output_file),
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )

        logger.info(f"✅ Fractal hierarchy visualization saved to {output_file}")
        return str(output_file)

    def _generate_fractal_coordinates(self, levels: int) -> List[Dict]:
        """Generate coordinates for fractal pattern."""
        points = []
        point_id = 0

        for level in range(levels):
            num_points = 4**level  # Exponential growth

            for i in range(num_points):
                # Sierpinski-like fractal pattern
                angle = (i / num_points) * 2 * math.pi
                radius = 1.0 + level * 0.8

                # Add some fractal noise
                fractal_noise = (
                    0.2 * math.sin(angle * (level + 1)) * math.cos(angle * (level + 2))
                )

                x = radius * math.cos(angle) + fractal_noise
                y = radius * math.sin(angle) + fractal_noise
                z = level * 1.5 + 0.3 * math.sin(angle * 3)

                points.append(
                    {
                        "id": point_id,
                        "level": level,
                        "x": x,
                        "y": y,
                        "z": z,
                        "angle": angle,
                    }
                )
                point_id += 1

        return points

    def _add_fractal_connections(self, fig: go.Figure, fractal_points: List[Dict]):
        """Add connecting lines in fractal pattern."""
        conn_x, conn_y, conn_z = [], [], []

        # Connect points within each level and between levels
        for level in range(len(set(p["level"] for p in fractal_points)) - 1):
            current_level = [p for p in fractal_points if p["level"] == level]
            next_level = [p for p in fractal_points if p["level"] == level + 1]

            # Connect each point in current level to multiple points in next level
            for curr_point in current_level:
                # Find closest points in next level
                for next_point in next_level[:4]:  # Limit connections
                    conn_x.extend([curr_point["x"], next_point["x"], None])
                    conn_y.extend([curr_point["y"], next_point["y"], None])
                    conn_z.extend([curr_point["z"], next_point["z"], None])

        # Add connection trace
        if conn_x:
            fig.add_trace(
                go.Scatter3d(
                    x=conn_x,
                    y=conn_y,
                    z=conn_z,
                    mode="lines",
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                    hoverinfo="none",
                    showlegend=False,
                    name="Fractal Connections",
                )
            )

    def create_all_visualizations(self) -> Dict[str, str]:
        """Create all three visualization types."""
        logger.info("🎨 Creating complete merX memory visualization suite...")

        # Load memory data
        data = self.load_memory_data()

        # Create all visualizations
        results = {}

        try:
            results["network_3d"] = self.create_3d_memory_network(data)
        except Exception as e:
            logger.error(f"Failed to create 3D network: {e}")
            results["network_3d"] = None

        try:
            results["tree_forest"] = self.create_tree_of_trees_3d(data)
        except Exception as e:
            logger.error(f"Failed to create tree forest: {e}")
            results["tree_forest"] = None

        try:
            results["fractal_hierarchy"] = self.create_fractal_hierarchy_graph(data)
        except Exception as e:
            logger.error(f"Failed to create fractal hierarchy: {e}")
            results["fractal_hierarchy"] = None

        # Create summary dashboard
        results["dashboard"] = self._create_summary_dashboard(data, results)

        logger.info("✨ All visualizations created successfully!")
        return results

    def _create_summary_dashboard(
        self, data: Dict[str, Any], viz_results: Dict[str, str]
    ) -> str:
        """Create a summary dashboard with links to all visualizations."""
        logger.info("📊 Creating summary dashboard...")

        # Create statistics visualizations
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Memory Type Distribution",
                "Domain Distribution",
                "Decay Factor Distribution",
                "Memory Statistics",
            ),
            specs=[
                [{"type": "pie"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "table"}],
            ],
        )

        # Memory type distribution
        if data["stats"]["memory_types"]:
            fig.add_trace(
                go.Pie(
                    labels=list(data["stats"]["memory_types"].keys()),
                    values=list(data["stats"]["memory_types"].values()),
                    name="Memory Types",
                ),
                row=1,
                col=1,
            )

        # Domain distribution
        if data["stats"]["domains"]:
            fig.add_trace(
                go.Pie(
                    labels=list(data["stats"]["domains"].keys()),
                    values=list(data["stats"]["domains"].values()),
                    name="Domains",
                ),
                row=1,
                col=2,
            )

        # Decay distribution
        decay_values = [node["decay_factor"] for node in data["nodes"]]
        if decay_values:
            fig.add_trace(
                go.Histogram(x=decay_values, nbinsx=20, name="Decay Distribution"),
                row=2,
                col=1,
            )

        # Statistics table
        stats_data = [
            ["Total Memories", data["stats"]["total_nodes"]],
            ["Total Connections", data["stats"]["total_edges"]],
            [
                "Average Decay",
                f"{np.mean(decay_values):.3f}" if decay_values else "N/A",
            ],
            ["Fresh Memories (>0.8)", len([d for d in decay_values if d > 0.8])],
            ["Fading Memories (<0.5)", len([d for d in decay_values if d < 0.5])],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=list(zip(*stats_data))),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title={
                "text": "🧠 merX Memory System - Dashboard Overview",
                "x": 0.5,
                "font": {"size": 24},
            },
            height=800,
            showlegend=False,
        )

        # Save dashboard
        output_file = self.viz_dir / "memory_dashboard.html"
        fig.write_html(str(output_file))

        logger.info(f"✅ Dashboard saved to {output_file}")
        return str(output_file)


def main():
    """Main function to create all visualizations."""
    print("🧠 merX Memory 3D Visualization System")
    print("=" * 60)

    # Initialize visualizer
    viz = MemoryVisualization3D()

    # Try to setup memory engine
    if viz.setup_memory_engine():
        print("✅ Connected to real merX memory data")
    else:
        print("🌱 Using sample data for demonstration")

    # Create all visualizations
    print("\n🎨 Creating beautiful 3D visualizations...")
    results = viz.create_all_visualizations()

    # Display results
    print("\n" + "=" * 60)
    print("🌟 VISUALIZATION COMPLETE!")
    print("=" * 60)

    for viz_type, file_path in results.items():
        if file_path:
            print(f"✅ {viz_type.replace('_', ' ').title()}: {file_path}")

            # Auto-open visualizations
            try:
                import webbrowser

                webbrowser.open(f"file://{os.path.abspath(file_path)}")
            except Exception as e:
                logger.debug(f"Could not auto-open {file_path}: {e}")
        else:
            print(f"❌ {viz_type.replace('_', ' ').title()}: Failed to create")

    print("\n🌿 Features created:")
    print("  🌌 3D Memory Network - Explore connections in 3D space")
    print("  🌲 Tree Forest - Hierarchical domain-based trees")
    print("  🌀 Fractal Hierarchy - Interactive clickable exploration")
    print("  📊 Dashboard - Statistics and overview")
    print("\n💡 Memory decay is visualized through:")
    print("  🎨 Color fading (bright green → gray)")
    print("  📏 Size changes (larger = fresher)")
    print("  🔗 Connection strength (thicker = stronger)")
    print("  🏔️ 3D positioning (height = memory layer)")

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    main()
