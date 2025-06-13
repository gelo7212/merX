#!/usr/bin/env python3
"""
Enhanced Database Viewer for merX Memory System

This enhanced script provides both standalone visualizations and a Streamlit dashboard
for exploring the memory database structure, connections, and search results.

NEW FEATURES:
- Interactive 3D network graph visualization
- Enhanced UI with better interactivity
- Improved performance for large datasets
- Beautiful styling and animations
"""

import os
import sys
import json
import logging
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from uuid import UUID
import numpy as np

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
from src.container.enhanced_di_container import create_enhanced_container

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUID objects and other non-serializable types."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        elif hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):  # Custom objects
            return str(obj)
        return super().default(obj)


def clean_data_for_json(data):
    """Clean data structure to make it JSON serializable."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Convert UUID keys to strings
            clean_key = str(key) if isinstance(key, UUID) else key
            cleaned[clean_key] = clean_data_for_json(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, UUID):
        return str(data)
    elif hasattr(data, "isoformat"):  # datetime objects
        return data.isoformat()
    elif hasattr(data, "__dict__") and not isinstance(data, (str, int, float, bool)):
        # Custom objects that aren't basic types
        return str(data)
    else:
        return data


class EnhancedDatabaseViewer:
    """Enhanced visualizes memory database structure and relationships with 3D capabilities."""

    def __init__(self):
        self.engine = None
        self.graph = nx.Graph()
        self.viz_dir = "data/visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)

    def setup_viewer(self, data_dir: str = "data/test_output"):
        """Set up the database viewer with memory engine."""
        logger.info("Setting up enhanced database viewer...")

        try:
            # Try to load existing memory data in order of preference
            data_sources = [
                f"{data_dir}/hp_mini.mex",  # Test output data (tree-100 or other tests)
                "data/test_output/hp_mini.mex",  # Search accuracy test data
                "data/test_output/hp_mini.mex",  # Extreme performance test data
                "data/temp_viewer_memory.mex",  # Fallback empty memory
            ]

            loaded = False
            for data_path in data_sources:
                if os.path.exists(data_path):
                    try:
                        self.engine = EnhancedMemoryEngineFactory.create_engine(
                            data_path=data_path
                        )
                        logger.info(f"✅ Loaded memory data from {data_path}")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load data from {data_path}: {e}")
                        continue

            if not loaded:
                logger.warning(
                    "No existing memory data found, creating empty memory engine"
                )
                self.engine = EnhancedMemoryEngineFactory.create_engine(
                    data_path="data/temp_viewer_memory.mex"
                )

        except Exception as e:
            logger.error(f"Failed to setup database viewer: {e}")
            raise

    def load_memory_data(self) -> Dict[str, Any]:
        """Load and analyze memory data."""
        logger.info("Loading memory data for analysis...")

        data = {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes": 0,
                "total_edges": 0,
                "domains": {},
                "node_types": {},
                "tag_distribution": {},
            },
        }
        try:
            # Get all nodes from storage
            if hasattr(self.engine.storage, "ramx") and self.engine.storage.ramx:
                ramx_nodes = self.engine.storage.ramx.get_all_nodes()

                # Handle both dict and list returns
                if isinstance(ramx_nodes, dict):
                    node_items = ramx_nodes.items()
                elif isinstance(ramx_nodes, list):
                    node_items = [(str(i), node) for i, node in enumerate(ramx_nodes)]
                else:
                    logger.warning(f"Unexpected ramx_nodes type: {type(ramx_nodes)}")
                    node_items = []

                for node_id, node in node_items:
                    # Extract node information
                    node_data = {
                        "id": str(node_id),
                        "content": (
                            getattr(node, "content", "")[:100] + "..."
                            if len(getattr(node, "content", "")) > 100
                            else getattr(node, "content", "")
                        ),
                        "full_content": getattr(node, "content", ""),
                        "node_type": getattr(node, "node_type", "unknown"),
                        "tags": getattr(node, "tags", []),
                        "created_at": (
                            getattr(node, "created_at", datetime.now()).isoformat()
                            if hasattr(node, "created_at")
                            else datetime.now().isoformat()
                        ),
                        "activation_count": getattr(node, "activation_count", 0),
                        "links": getattr(node, "links", {}),  # Extract actual links
                    }

                    data["nodes"].append(node_data)

                    # Extract real edges from node links
                    node_links = getattr(node, "links", {})
                    if isinstance(node_links, dict):
                        for target_id, link_data in node_links.items():
                            if (
                                isinstance(link_data, (tuple, list))
                                and len(link_data) >= 2
                            ):
                                weight, link_type = link_data[0], link_data[1]
                            else:
                                weight, link_type = 1.0, "default"
                            data["edges"].append(
                                {
                                    "source": str(node_id),
                                    "target": str(target_id),
                                    "weight": weight,
                                    "link_type": link_type,
                                }
                            )

                    # Update statistics
                    node_type = node_data["node_type"]
                    data["stats"]["node_types"][node_type] = (
                        data["stats"]["node_types"].get(node_type, 0) + 1
                    )

                    for tag in node_data["tags"]:
                        data["stats"]["tag_distribution"][tag] = (
                            data["stats"]["tag_distribution"].get(tag, 0) + 1
                        )                        
                        
                        # Track domains (assume first tag is domain)
                        if tag in [
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
                        ]:
                            data["stats"]["domains"][tag] = (
                                data["stats"]["domains"].get(tag, 0) + 1
                            )

                data["stats"]["total_nodes"] = len(data["nodes"])
                data["stats"]["total_edges"] = len(data["edges"])
        except Exception as e:
            logger.error(f"Error loading memory data: {e}")
            # Generate sample data for demonstration
            self._generate_sample_data(data)

        logger.info(
            f"✅ Loaded {data['stats']['total_nodes']} nodes and {data['stats']['total_edges']} edges"
        )
        return data

    def _generate_sample_data(self, data: Dict[str, Any]):
        """Generate sample data for demonstration when no real data available."""
        logger.info("Generating sample data for demonstration...")        
        
        domains = [
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

        # Generate sample nodes
        for i in range(50):
            domain = domains[i % len(domains)]
            data["nodes"].append(
                {
                    "id": f"node_{i}",
                    "content": f"Sample content for {domain} node {i}",
                    "full_content": f"This is detailed sample content for {domain} node {i}. It contains information about various concepts and their relationships.",
                    "node_type": "concept",
                    "tags": [domain, f"topic_{i%5}", "sample"],
                    "created_at": datetime.now().isoformat(),
                    "activation_count": i % 10,
                }
            )

            data["stats"]["domains"][domain] = (
                data["stats"]["domains"].get(domain, 0) + 1
            )
            data["stats"]["node_types"]["concept"] = (
                data["stats"]["node_types"].get("concept", 0) + 1
            )

        # Generate sample edges
        nodes = data["nodes"]
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1 :], i + 1):
                shared_tags = set(node1["tags"]) & set(node2["tags"])
                if shared_tags and len(shared_tags) > 1:  # At least 2 shared tags
                    data["edges"].append(
                        {
                            "source": node1["id"],
                            "target": node2["id"],
                            "weight": len(shared_tags),
                            "link_type": "similarity",
                        }
                    )

        data["edges"] = data["edges"][:100]  # Limit for performance
        data["stats"]["total_nodes"] = len(data["nodes"])
        data["stats"]["total_edges"] = len(data["edges"])

    def create_enhanced_2d_graph(self, data: Dict[str, Any]) -> str:
        """Create an enhanced interactive 2D network graph visualization."""
        logger.info("Creating enhanced 2D network graph visualization...")

        # Increased limit to show more connections and network structure
        max_nodes_for_viz = 100000  # Higher limit to show full network connections
        nodes_to_use = data["nodes"]
        edges_to_use = data["edges"]

        if len(nodes_to_use) > max_nodes_for_viz:
            logger.info(
                f"Sampling {max_nodes_for_viz} nodes from {len(nodes_to_use)} for visualization performance"
            )
            # Sample nodes by activation count (keep most active ones)
            sorted_nodes = sorted(
                nodes_to_use, key=lambda x: x.get("activation_count", 0), reverse=True
            )
            nodes_to_use = sorted_nodes[:max_nodes_for_viz]

            # Filter edges to only include those between sampled nodes
            node_ids = {node["id"] for node in nodes_to_use}
            edges_to_use = [
                edge
                for edge in edges_to_use
                if edge["source"] in node_ids and edge["target"] in node_ids
            ]

            logger.info(
                f"Using {len(nodes_to_use)} nodes and {len(edges_to_use)} edges for visualization"
            )

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in nodes_to_use:
            G.add_node(
                node["id"],
                content=node["content"],
                node_type=node["node_type"],
                tags=node["tags"],
                activation_count=node["activation_count"],
            )

        # Add edges
        for edge in edges_to_use:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

        # Calculate layout - use faster algorithms for large graphs
        if len(G.nodes()) > 0:
            try:
                if len(G.nodes()) > 1000:
                    # Use faster layout algorithm for large graphs
                    logger.info("Using faster layout algorithm for large graph...")
                    pos = nx.spring_layout(G, k=2, iterations=20)
                elif len(G.nodes()) > 500:
                    # Medium-sized graphs
                    pos = nx.spring_layout(G, k=1.5, iterations=30)
                else:
                    # Small graphs can use more iterations
                    pos = nx.spring_layout(G, k=1, iterations=50)
            except Exception as e:
                logger.warning(f"Layout calculation failed: {e}, using grid layout")
                pos = {node: (i % 50, i // 50) for i, node in enumerate(G.nodes())}
        else:
            pos = {}

        # Prepare data for Plotly with enhanced interactivity
        # Create single edge trace for better performance
        edge_x = []
        edge_y = []
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="rgba(125, 125, 125, 0.5)"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        # Create enhanced node trace with better interactivity
        if pos:
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]

            # Enhanced hover text with more information
            node_text = []
            node_colors = []
            node_sizes = []

            for node in G.nodes():
                node_data = G.nodes[node]
                # Create rich hover text
                hover_text = (
                    f"<b>🔍 Node ID:</b> {node}<br>"
                    f"<b>📁 Type:</b> {node_data.get('node_type', 'Unknown')}<br>"
                    f"<b>⚡ Activation:</b> {node_data.get('activation_count', 0)}<br>"
                    f"<b>🔗 Connections:</b> {G.degree(node)}<br>"
                    f"<b>🏷️ Tags:</b> {', '.join(node_data.get('tags', []))}<br>"
                    f"<b>📝 Content:</b><br>{node_data.get('content', 'No content')}"
                )
                node_text.append(hover_text)

                # Color by activation count
                activation = node_data.get("activation_count", 0)
                node_colors.append(activation)

                # Size by degree (number of connections)
                degree = G.degree(node)
                node_sizes.append(max(8, min(25, 8 + degree * 2)))

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                hovertext=node_text,
                marker=dict(
                    showscale=True,
                    colorscale="Viridis",
                    color=node_colors,
                    size=node_sizes,
                    colorbar=dict(
                        thickness=15, len=0.5, x=1.02, title="Activation Count"
                    ),
                    line=dict(width=2, color="white"),
                ),
                showlegend=False,
            )
        else:
            node_trace = None

        # Create enhanced figure with better layout and controls
        traces = [edge_trace] if edge_trace else []
        if node_trace is not None:
            traces.append(node_trace)

        if not traces:
            fig = go.Figure()
            fig.add_annotation(
                text="No memory data available for visualization",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(title="Memory Database Network Graph - No Data")
        else:
            fig = go.Figure(
                data=traces,
                layout=go.Layout(
                    title=dict(
                        text="🧠 Enhanced Interactive Memory Database Network Graph (2D)",
                        font=dict(size=20),
                        x=0.5,
                    ),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=40, l=40, r=40, t=80),
                    annotations=[
                        dict(
                            text="💡 Hover over nodes for details • Drag to pan • Scroll to zoom • Click toolbar for more options",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.1,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(color="#666", size=12),
                        )
                    ],
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(0,0,0,0.1)",
                        zeroline=False,
                        showticklabels=False,
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(0,0,0,0.1)",
                        zeroline=False,
                        showticklabels=False,
                    ),
                    plot_bgcolor="rgba(248,249,250,1)",
                    # Enhanced interactivity
                    dragmode="pan",
                ),
            )  # Save the graph
        output_file = f"{self.viz_dir}/enhanced_2d_graph.html"
        fig.write_html(
            output_file, config={"displayModeBar": True, "displaylogo": False}
        )
        logger.info(f"✅ Enhanced 2D network graph saved to {output_file}")

        return output_file

    def create_3d_network_graph(self, data: Dict[str, Any]) -> str:
        """Create an interactive 3D network graph visualization with advanced features."""
        logger.info(
            "🌌 Creating advanced interactive 3D network graph visualization..."
        )

        # Increased limit for 3D to show more connections
        max_nodes_for_3d = 100000  # Higher limit to show rich network connections in 3D
        nodes_to_use = data["nodes"]
        edges_to_use = data["edges"]

        if len(nodes_to_use) > max_nodes_for_3d:
            logger.info(
                f"Sampling {max_nodes_for_3d} nodes from {len(nodes_to_use)} for 3D visualization performance"
            )
            # Sample nodes by activation count (keep most active ones)
            sorted_nodes = sorted(
                nodes_to_use, key=lambda x: x.get("activation_count", 0), reverse=True
            )
            nodes_to_use = sorted_nodes[:max_nodes_for_3d]

            # Filter edges to only include those between sampled nodes
            node_ids = {node["id"] for node in nodes_to_use}
            edges_to_use = [
                edge
                for edge in edges_to_use
                if edge["source"] in node_ids and edge["target"] in node_ids
            ]

            logger.info(
                f"Using {len(nodes_to_use)} nodes and {len(edges_to_use)} edges for 3D visualization"
            )

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes with enhanced attributes
        for node in nodes_to_use:
            G.add_node(
                node["id"],
                content=node["content"],
                node_type=node["node_type"],
                tags=node["tags"],
                activation_count=node["activation_count"],
            )

        # Add edges
        for edge in edges_to_use:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

        # Calculate 3D layout
        if len(G.nodes()) > 0:
            try:
                if len(G.nodes()) > 800:
                    logger.info("Using faster 3D layout for large graph...")
                    # Use 3D spring layout with reduced iterations for performance
                    pos = nx.spring_layout(G, dim=3, k=3, iterations=15)
                else:
                    # Standard 3D layout
                    pos = nx.spring_layout(G, dim=3, k=2, iterations=30)
            except Exception as e:
                logger.warning(
                    f"3D layout calculation failed: {e}, using 3D grid layout"
                )
                # Create 3D grid layout as fallback
                nodes = list(G.nodes())
                grid_size = int(len(nodes) ** (1 / 3)) + 1
                pos = {}
                for i, node in enumerate(nodes):
                    x = i % grid_size
                    y = (i // grid_size) % grid_size
                    z = i // (grid_size * grid_size)
                    pos[node] = (x, y, z)
        else:
            pos = {}

        # Create enhanced 3D visualization with interactive features
        return self._create_advanced_3d_graph(G, pos, data, edges_to_use)    
    
    def _create_advanced_3d_graph(self, G, pos, data, edges_to_use):
        """Create advanced 3D graph with connection toggles and path highlighting - OPTIMIZED VERSION."""

        # Create multiple edge traces for different connection types
        edge_traces = {}
        connection_types = set()

        # Separate edges by type for toggle functionality
        for edge in edges_to_use:
            edge_type = edge.get("link_type", "default")
            connection_types.add(edge_type)
            if edge_type not in edge_traces:
                edge_traces[edge_type] = {"x": [], "y": [], "z": [], "edges": []}

        # Build edge traces with optimized lookup
        edge_lookup = {}
        for orig_edge in edges_to_use:
            key1 = (orig_edge["source"], orig_edge["target"])
            key2 = (orig_edge["target"], orig_edge["source"])
            edge_type = orig_edge.get("link_type", "default")
            edge_lookup[key1] = edge_type
            edge_lookup[key2] = edge_type

        # Build edge traces
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]

                # Find edge type from lookup
                edge_type = edge_lookup.get(edge, "default")

                if edge_type in edge_traces:
                    edge_traces[edge_type]["x"].extend([x0, x1, None])
                    edge_traces[edge_type]["y"].extend([y0, y1, None])
                    edge_traces[edge_type]["z"].extend([z0, z1, None])
                    edge_traces[edge_type]["edges"].append((edge[0], edge[1]))

        # Create Plotly traces for different connection types
        plotly_traces = []

        # Color scheme for different connection types
        connection_colors = {
            "similarity": "rgba(255, 100, 100, 0.6)",  # Red
            "temporal": "rgba(100, 255, 100, 0.6)",  # Green
            "semantic": "rgba(100, 100, 255, 0.6)",  # Blue
            "causal": "rgba(255, 255, 100, 0.6)",  # Yellow
            "default": "rgba(125, 125, 125, 0.6)",  # Gray
        }

        for edge_type, trace_data in edge_traces.items():
            if trace_data["x"]:  # Only create trace if there are edges
                color = connection_colors.get(edge_type, "rgba(125, 125, 125, 0.6)")
                edge_trace = go.Scatter3d(
                    x=trace_data["x"],
                    y=trace_data["y"],
                    z=trace_data["z"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="none",
                    name=f"{edge_type.title()} Connections",
                    visible=True,  # Start with all connections visible
                    legendgroup="connections",
                )
                plotly_traces.append(edge_trace)

        # Create enhanced node trace with click interactions
        if pos:
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_z = [pos[node][2] for node in G.nodes()]

            # Enhanced hover text and styling
            node_text = []
            node_colors = []
            node_sizes = []
            node_ids = []

            for node in G.nodes():
                node_data = G.nodes[node]
                node_ids.append(node)                # Rich hover information with safe string handling
                neighbors = list(G.neighbors(node))
                neighbor_info = f"Connected to: {', '.join(str(n)[:50] for n in neighbors[:5])}"  # Limit neighbor ID length
                if len(neighbors) > 5:
                    neighbor_info += f" (+{len(neighbors)-5} more)"

                # Safe content truncation
                content_preview = str(node_data.get('content', 'No content'))[:150].replace('\n', ' ').replace('\r', ' ')
                tags_preview = ', '.join(str(tag)[:20] for tag in node_data.get('tags', [])[:5])  # Limit tags
                
                hover_text = (
                    f"<b>🔍 Node ID:</b> {str(node)[:50]}<br>"
                    f"<b>📁 Type:</b> {str(node_data.get('node_type', 'Unknown'))[:20]}<br>"
                    f"<b>⚡ Activation:</b> {node_data.get('activation_count', 0)}<br>"
                    f"<b>🏷️ Tags:</b> {tags_preview}<br>"
                    f"<b>🔗 Connections:</b> {G.degree(node)}<br>"
                    f"<b>🌐 {neighbor_info}</b><br>"
                    f"<b>📝 Content:</b><br>{content_preview}..."
                    f"<br><br><i>💡 Click to highlight paths</i>"
                )
                node_text.append(hover_text)

                # Color mapping by node type and activation
                activation = node_data.get("activation_count", 0)
                node_type = node_data.get("node_type", "unknown")

                # Create color based on both type and activation
                type_colors = {
                    "concept": 0.8,
                    "fact": 0.6,
                    "opinion": 0.4,
                    "reference": 0.3,
                    "note": 0.2,
                    "experience": 0.1,
                    "unknown": 0.0,
                }
                base_color = type_colors.get(node_type, 0.0)
                color_value = base_color + (activation / 100.0) * 0.5
                node_colors.append(color_value)

                # Size based on degree centrality and activation
                degree = G.degree(node)
                size = max(8, min(30, 10 + degree * 2 + activation * 0.5))
                node_sizes.append(size)

            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(
                        title="Node Type & Activation", thickness=20, len=0.7, x=1.02
                    ),
                    line=dict(color="white", width=1),
                    opacity=0.9,
                ),
                text=node_text,
                hoverinfo="text",
                name="Memory Nodes",
                customdata=node_ids,  # Store node IDs for click events
                legendgroup="nodes",
            )
            plotly_traces.append(node_trace)

        # Create enhanced 3D figure with interactive controls
        fig = go.Figure(
            data=plotly_traces,
            layout=go.Layout(
                title=dict(
                    text="🌌 Advanced Interactive 3D Memory Network Graph",
                    font=dict(size=22),
                    x=0.5,
                ),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    x=1.02,
                    y=1,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
                margin=dict(l=40, r=200, b=40, t=80),  # More space for legend
                scene=dict(
                    xaxis=dict(
                        title="Memory Space X",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.3)",
                        showbackground=True,
                        backgroundcolor="rgba(230, 230, 250, 0.1)",
                    ),
                    yaxis=dict(
                        title="Memory Space Y",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.3)",
                        showbackground=True,
                        backgroundcolor="rgba(230, 250, 230, 0.1)",
                    ),
                    zaxis=dict(
                        title="Memory Space Z",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.3)",
                        showbackground=True,
                        backgroundcolor="rgba(250, 230, 230, 0.1)",
                    ),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    bgcolor="rgba(0, 0, 0, 0.9)",
                    aspectmode="cube",
                ),
                annotations=[
                    dict(
                        text="🎮 Drag to rotate • Scroll to zoom • Click nodes for path highlighting • Toggle connections in legend",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.02,
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(color="#888", size=14),
                    )
                ],
                paper_bgcolor="rgba(240, 240, 240, 1)",
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list(
                            [
                                dict(
                                    args=[{"visible": [True] * len(plotly_traces)}],
                                    label="Show All Connections",
                                    method="restyle",
                                ),
                                dict(
                                    args=[
                                        {
                                            "visible": [
                                                (
                                                    False
                                                    if "Connections" in trace.name
                                                    else True
                                                )
                                                for trace in plotly_traces
                                            ]
                                        }
                                    ],
                                    label="Hide All Connections",
                                    method="restyle",
                                ),
                                dict(
                                    args=[
                                        {
                                            "visible": [
                                                (
                                                    True
                                                    if "similarity"
                                                    in trace.name.lower()
                                                    or "Nodes" in trace.name
                                                    else False
                                                )
                                                for trace in plotly_traces
                                            ]
                                        }
                                    ],
                                    label="Similarity Only",
                                    method="restyle",
                                ),
                                dict(
                                    args=[
                                        {
                                            "visible": [
                                                (
                                                    True
                                                    if "semantic" in trace.name.lower()
                                                    or "Nodes" in trace.name
                                                    else False
                                                )
                                                for trace in plotly_traces
                                            ]
                                        }
                                    ],
                                    label="Semantic Only",
                                    method="restyle",
                                ),
                            ]
                        ),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.02,
                        yanchor="top",
                    ),
                ],
            ),
        )        # Add JavaScript for click interactions and path highlighting
        # Create safe graph data structure to prevent circular references
        safe_graph_data = {}
        try:
            for node_id in G.nodes():
                neighbors = list(G.neighbors(node_id))
                # Convert to strings to prevent JSON issues
                safe_graph_data[str(node_id)] = [str(n) for n in neighbors[:10]]  # Limit neighbors to prevent overflow
        except Exception as e:
            logger.warning(f"Error creating graph data: {e}")
            safe_graph_data = {}

        click_script = (
            """
        <script>
        var plotDiv = document.getElementById('plotly-div');
        var selectedNode = null;
        var originalColors = [];
        var originalSizes = [];
        var graphData = """
            + json.dumps(safe_graph_data)
            + """;
        
        // Store original node properties safely
        function storeOriginalProperties() {
            try {
                var nodeTrace = plotDiv.data.find(trace => trace.name === "Memory Nodes");
                if (nodeTrace && nodeTrace.marker) {
                    originalColors = nodeTrace.marker.color ? [...nodeTrace.marker.color] : [];
                    originalSizes = nodeTrace.marker.size ? [...nodeTrace.marker.size] : [];
                }
            } catch (e) {
                console.warn("Error storing original properties:", e);
                originalColors = [];
                originalSizes = [];
            }
        }
        
        // Safe path finding with depth limit and cycle detection
        function findPath(start, end, maxDepth = 2) {
            if (!start || !end || start === end) return [start];
            
            try {
                var queue = [[String(start)]];
                var visited = new Set([String(start)]);
                var iterations = 0;
                var maxIterations = 100; // Prevent infinite loops
                
                while (queue.length > 0 && iterations < maxIterations) {
                    iterations++;
                    var path = queue.shift();
                    var node = path[path.length - 1];
                    
                    if (path.length > maxDepth) continue;
                    
                    var neighbors = graphData[node] || [];
                    for (var i = 0; i < Math.min(neighbors.length, 5); i++) { // Limit neighbors checked
                        var neighbor = String(neighbors[i]);
                        if (neighbor === String(end)) {
                            return [...path, neighbor];
                        }
                        
                        if (!visited.has(neighbor) && path.length < maxDepth) {
                            visited.add(neighbor);
                            queue.push([...path, neighbor]);
                        }
                    }
                }
                return null;
            } catch (e) {
                console.warn("Error in findPath:", e);
                return null;
            }
        }          // Enhanced connection highlighting - shows immediate connections and multi-hop paths
        function highlightPath(nodeId) {
            try {
                if (!nodeId) return;
                
                nodeId = String(nodeId); // Ensure string type
                
                if (selectedNode === nodeId) {
                    // Deselect - restore original appearance
                    resetHighlight();
                    selectedNode = null;
                    return;
                }
                
                selectedNode = nodeId;
                var nodeTrace = plotDiv.data.find(trace => trace.name === "Memory Nodes");
                
                if (!nodeTrace || !nodeTrace.customdata || originalColors.length === 0) {
                    console.warn("Node trace or original data not found");
                    return;
                }
                
                var newColors = [...originalColors];
                var newSizes = [...originalSizes];
                
                // Find all connected nodes at different levels
                var directConnections = graphData[nodeId] || [];
                var secondLevelConnections = new Set();
                
                // Get second-level connections (connections of connections)
                for (var j = 0; j < Math.min(directConnections.length, 10); j++) {
                    var directNeighbor = String(directConnections[j]);
                    var neighborConnections = graphData[directNeighbor] || [];
                    for (var k = 0; k < Math.min(neighborConnections.length, 5); k++) {
                        var secondLevel = String(neighborConnections[k]);
                        if (secondLevel !== nodeId && !directConnections.includes(secondLevel)) {
                            secondLevelConnections.add(secondLevel);
                        }
                    }
                }
                
                // Find node index safely
                var nodeIndex = -1;
                for (var i = 0; i < nodeTrace.customdata.length; i++) {
                    if (String(nodeTrace.customdata[i]) === nodeId) {
                        nodeIndex = i;
                        break;
                    }
                }                
                if (nodeIndex !== -1 && nodeIndex < newColors.length) {
                    // Highlight selected node with bright color
                    newColors[nodeIndex] = 2.5; // Brightest color
                    newSizes[nodeIndex] = Math.min(60, (originalSizes[nodeIndex] || 15) * 2.0);
                    
                    // Color nodes by connection level
                    for (var i = 0; i < nodeTrace.customdata.length; i++) {
                        var currentNodeId = String(nodeTrace.customdata[i]);
                        
                        if (currentNodeId === nodeId) continue; // Skip selected node
                        
                        // Check if it's a direct connection
                        if (directConnections.includes(currentNodeId)) {
                            newColors[i] = 1.8; // Bright for direct connections
                            newSizes[i] = Math.min(45, (originalSizes[i] || 12) * 1.5);
                        }
                        // Check if it's a second-level connection
                        else if (secondLevelConnections.has(currentNodeId)) {
                            newColors[i] = 1.2; // Medium for second-level
                            newSizes[i] = Math.min(35, (originalSizes[i] || 10) * 1.2);
                        }
                        // Dim unconnected nodes
                        else {
                            newColors[i] = Math.max(0.1, (originalColors[i] || 0.5) * 0.3);
                            newSizes[i] = Math.max(5, (originalSizes[i] || 10) * 0.7);
                        }
                    }
                }                
                // Safely update the plot
                var nodeTraceIndex = plotDiv.data.findIndex(trace => trace.name === "Memory Nodes");
                if (nodeTraceIndex !== -1) {
                    Plotly.restyle(plotDiv, {
                        'marker.color': [newColors],
                        'marker.size': [newSizes]
                    }, [nodeTraceIndex]);
                }
                
        // Enhanced status bar to show connection information
        function updateStatus(selectedNodeId, directConnections, secondLevelConnections) {
            var statusElement = document.getElementById('status');
            if (statusElement && selectedNodeId) {
                var totalConnections = directConnections.length + secondLevelConnections.size;
                statusElement.innerHTML = 
                    `<strong>Selected:</strong> ${selectedNodeId.substring(0, 20)}... | ` +
                    `<strong style="color: #ff6b6b;">Direct:</strong> ${directConnections.length} | ` +
                    `<strong style="color: #4ecdc4;">2nd Level:</strong> ${secondLevelConnections.size} | ` +
                    `<strong style="color: #45b7d1;">Total Network:</strong> ${totalConnections} | ` +
                    `<strong>Click another node or same node to deselect</strong>`;
            } else if (statusElement) {
                statusElement.innerHTML = '<strong>Status:</strong> Click any node to highlight its connections and explore the network paths';
            }
        }                
                // Update status with connection details
                updateStatus(nodeId, directConnections, Array.from(secondLevelConnections));
            } catch (e) {
                console.error("Error in highlightPath:", e);
                resetHighlight();
            }
        }        
        // Reset all highlights safely
        function resetHighlight() {
            try {
                if (originalColors.length === 0 || originalSizes.length === 0) {
                    console.warn("Original properties not available for reset");
                    return;
                }
                
                var nodeTraceIndex = plotDiv.data.findIndex(trace => trace.name === "Memory Nodes");
                if (nodeTraceIndex !== -1) {
                    Plotly.restyle(plotDiv, {
                        'marker.color': [originalColors],
                        'marker.size': [originalSizes]
                    }, [nodeTraceIndex]);
                }
                
                updateStatus(null, [], []);
            } catch (e) {
                console.error("Error in resetHighlight:", e);
            }
        }
        
        // Handle click events safely
        function handlePlotClick(data) {
            try {
                if (data && data.points && data.points.length > 0) {
                    var point = data.points[0];
                    if (point && point.customdata) {
                        highlightPath(point.customdata);
                    }
                }
            } catch (e) {
                console.error("Error handling click:", e);
            }
        }
        
        // Safe initialization
        function initializeGraph() {
            try {
                if (plotDiv) {
                    // Remove any existing event listeners to prevent duplication
                    plotDiv.removeAllListeners && plotDiv.removeAllListeners('plotly_click');
                    plotDiv.removeAllListeners && plotDiv.removeAllListeners('plotly_afterplot');
                    
                    // Add new event listeners
                    plotDiv.on('plotly_click', handlePlotClick);
                    plotDiv.on('plotly_afterplot', storeOriginalProperties);
                    
                    // Initial property storage
                    setTimeout(storeOriginalProperties, 1000); // Delay to ensure plot is ready
                }
            } catch (e) {
                console.error("Error initializing graph:", e);
            }
        }
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeGraph);
        } else {
            initializeGraph();
        }
        </script>
        """
        )

        # Save the 3D graph with enhanced config and custom JavaScript
        output_file = f"{self.viz_dir}/memory_3d_graph.html"

        # Write HTML with custom JavaScript and status bar
        html_content = fig.to_html(
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["pan3d", "orbitRotation", "tableRotation"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "memory_network_3d",
                    "height": 800,
                    "width": 1200,
                    "scale": 2,
                },
            },
            div_id="plotly-div",
        )        # Add enhanced status bar with better styling
        status_bar = """
        <div id="status" style="
            position: fixed; 
            bottom: 0; 
            left: 0; 
            right: 0; 
            background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(40,40,40,0.9)); 
            color: white; 
            padding: 15px; 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 14px;
            text-align: center;
            z-index: 1000;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
            border-top: 2px solid #667eea;
        ">
            <strong>🎯 Interactive Network Explorer:</strong> Click any node to highlight its connections and explore the network paths
            <br><small style="opacity: 0.8;">💡 Direct connections shown in bright colors, 2nd-level connections in medium colors, unrelated nodes dimmed</small>
        </div>
        """

        # Insert custom JavaScript and status bar before closing body tag
        html_content = html_content.replace(
            "</body>", f"{status_bar}{click_script}</body>"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"✅ Advanced 3D network graph saved to {output_file}")
        return output_file

    def create_cluster_analysis_3d(self, data: Dict[str, Any]) -> str:
        """Create a 3D cluster analysis based on node types and domains."""
        logger.info("Creating 3D cluster analysis...")

        if not data["nodes"]:
            return self._create_empty_visualization("3D Cluster Analysis - No Data")

        # Group nodes by domain and type
        clusters = {}
        for node in data["nodes"]:
            domain = node.get("tags", ["unknown"])[0] if node.get("tags") else "unknown"
            node_type = node.get("node_type", "unknown")
            cluster_key = f"{domain}_{node_type}"

            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(node)

        # Create 3D visualization with clusters
        traces = []
        colors = px.colors.qualitative.Set3

        for i, (cluster_name, nodes) in enumerate(clusters.items()):
            if len(nodes) < 2:  # Skip clusters with too few nodes
                continue

            # Position nodes in 3D space based on cluster
            angle = 2 * np.pi * i / len(clusters)
            center_x = 5 * np.cos(angle)
            center_y = 5 * np.sin(angle)
            center_z = 0

            # Add some randomness within the cluster
            x_coords = [center_x + np.random.normal(0, 1) for _ in nodes]
            y_coords = [center_y + np.random.normal(0, 1) for _ in nodes]
            z_coords = [center_z + np.random.normal(0, 0.5) for _ in nodes]

            # Create hover text
            hover_texts = []
            for node in nodes:
                hover_text = (
                    f"<b>Cluster:</b> {cluster_name}<br>"
                    f"<b>Node:</b> {node['id']}<br>"
                    f"<b>Content:</b> {node['content'][:100]}..."
                )
                hover_texts.append(hover_text)

            # Create trace for this cluster
            trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker=dict(
                    size=[
                        max(5, min(15, node.get("activation_count", 0) + 5))
                        for node in nodes
                    ],
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(color="white", width=1),
                ),
                text=hover_texts,
                hoverinfo="text",
                name=cluster_name.replace("_", " ").title(),
            )
            traces.append(trace)

        if not traces:
            return self._create_empty_visualization("3D Cluster Analysis - No Clusters")

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text="🔬 3D Memory Cluster Analysis", font=dict(size=20), x=0.5
                ),
                scene=dict(
                    xaxis=dict(title="Domain Space X"),
                    yaxis=dict(title="Domain Space Y"),
                    zaxis=dict(title="Type Space Z"),
                    camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
                    bgcolor="rgba(0, 0, 0, 0.95)",
                ),
                margin=dict(l=40, r=40, b=40, t=80),
            ),
        )

        output_file = f"{self.viz_dir}/cluster_analysis_3d.html"
        fig.write_html(
            output_file, config={"displayModeBar": True, "displaylogo": False}
        )
        logger.info(f"✅ 3D cluster analysis saved to {output_file}")

        return output_file

    def _create_empty_visualization(self, title: str) -> str:
        """Create an empty visualization with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title=title)

        output_file = f"{self.viz_dir}/empty_viz.html"
        fig.write_html(output_file)
        return output_file

    def create_enhanced_dashboard(self) -> str:
        """Create an enhanced HTML dashboard with multiple visualizations."""
        logger.info("Creating enhanced HTML dashboard...")

        # Load data
        data = self.load_memory_data()

        # Create all visualizations
        viz_2d = self.create_enhanced_2d_graph(data)
        viz_3d = self.create_3d_network_graph(data)
        viz_cluster = self.create_cluster_analysis_3d(data)

        # Create dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 merX Enhanced Memory Database Viewer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }}
        .viz-card {{
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .viz-card h2 {{
            margin-top: 0;
            color: #444;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .nav-buttons {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
        }}
        .nav-button {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            font-weight: bold;
        }}
        .nav-button:hover {{
            background: white;
            color: #667eea;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 merX Enhanced Memory Database Viewer</h1>
        <p>Interactive exploration of memory structures with 2D & 3D visualizations</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{data['stats']['total_nodes']}</div>
            <div>Memory Nodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{data['stats']['total_edges']}</div>
            <div>Connections</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(data['stats']['domains'])}</div>
            <div>Knowledge Domains</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(data['stats']['node_types'])}</div>
            <div>Node Types</div>
        </div>
    </div>
    
    <div class="nav-buttons">
        <a href="#2d-view" class="nav-button">📊 Enhanced 2D View</a>
        <a href="#3d-view" class="nav-button">🌌 3D Network Graph</a>
        <a href="#cluster-view" class="nav-button">🔬 3D Cluster Analysis</a>
    </div>
    
    <div class="viz-grid">
        <div id="2d-view" class="viz-card">
            <h2>📊 Enhanced 2D Interactive Network Graph</h2>
            <p>Explore memory connections with enhanced interactivity. Hover over nodes for detailed information.</p>
            <iframe src="{os.path.basename(viz_2d)}"></iframe>
        </div>
        
        <div id="3d-view" class="viz-card">
            <h2>🌌 3D Interactive Memory Network</h2>
            <p>Experience your memory database in three dimensions. Click nodes to highlight connections. Use control buttons to toggle connection types.</p>
            <iframe src="{os.path.basename(viz_3d)}"></iframe>
        </div>
        
        <div id="cluster-view" class="viz-card">
            <h2>🔬 3D Cluster Analysis</h2>
            <p>Visualize how memories cluster by domain and type in 3D space.</p>
            <iframe src="{os.path.basename(viz_cluster)}"></iframe>
        </div>
    </div>
    
    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('.nav-button').forEach(button => {{
            button.addEventListener('click', function(e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                target.scrollIntoView({{ behavior: 'smooth' }});
            }});
        }});
    </script>
</body>
</html>
"""

        # Save dashboard
        dashboard_file = f"{self.viz_dir}/enhanced_dashboard.html"
        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(dashboard_html)

        logger.info(f"✅ Enhanced dashboard saved to {dashboard_file}")
        return dashboard_file

    def create_all_enhanced_visualizations(self):
        """Create all enhanced visualizations including 3D graphs."""
        logger.info("🚀 Creating all enhanced visualizations...")

        # Load memory data
        data = self.load_memory_data()

        # Create enhanced visualizations
        viz_files = []
        viz_files.append(self.create_enhanced_2d_graph(data))
        viz_files.append(self.create_3d_network_graph(data))
        viz_files.append(self.create_cluster_analysis_3d(data))
        dashboard_file = self.create_enhanced_dashboard()

        # Print beautiful summary
        print("\n" + "🌟" * 25)
        print("🧠 ENHANCED DATABASE VISUALIZATIONS CREATED")
        print("🌟" * 25)
        print(f"📊 Enhanced 2D Graph: {viz_files[0]}")
        print(f"🌌 3D Network Graph: {viz_files[1]}")
        print(f"🔬 3D Cluster Analysis: {viz_files[2]}")
        print(f"🎛️ Enhanced Dashboard: {dashboard_file}")
        print("🌟" * 25)
        print(
            "💡 Open the dashboard.html file in your browser for the best experience!"
        )
        print("🎮 Use mouse/trackpad to interact with 3D visualizations")
        print("🌟" * 25)

        return viz_files + [dashboard_file]


# Main execution functions
def main():
    """Main function to run the enhanced database viewer."""
    try:
        viewer = EnhancedDatabaseViewer()
        viewer.setup_viewer()

        print("\n🚀 Starting Enhanced Database Viewer...")
        print("=" * 50)

        # Create all enhanced visualizations
        viz_files = viewer.create_all_enhanced_visualizations()

        # Open dashboard automatically
        dashboard_file = viz_files[-1]  # Dashboard is the last file

        print(f"\n🎉 All visualizations created successfully!")
        print(f"📁 Files saved in: {viewer.viz_dir}")
        print(f"🌐 Opening dashboard: {dashboard_file}")

        # Try to open in browser
        try:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(dashboard_file)}")
            print("✅ Dashboard opened in browser!")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print(f"💡 Please manually open: {os.path.abspath(dashboard_file)}")

    except Exception as e:
        logger.error(f"Enhanced database viewer failed: {e}")
        raise


if __name__ == "__main__":
    main()
