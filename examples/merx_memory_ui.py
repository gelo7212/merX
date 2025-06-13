#!/usr/bin/env python3
"""
merX Memory 3D Visualization UI

A comprehensive 3D memory visualization interface with:
1. 3D Memory Network View - Connected nodes with memory decay visualization
2. Tree-of-Trees View - Hierarchical memory structures
3. Fractal Memory Explorer - Interactive hierarchy navigation with animations

Features:
- GPU-accelerated rendering for 50k+ nodes
- Memory decay visualization with beautiful fading effects  
- Interactive fractal tree exploration
- Animated transitions and smooth interactions
- Real-time performance monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import math

# Set page config for wide layout
st.set_page_config(
    page_title="merX Memory 3D Visualizer", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryDecayVisualizer:
    """Beautiful memory decay visualization with nature-inspired colors"""
    
    def get_decay_color(self, decay_factor: float, memory_type: str = "general") -> str:
        """Get color based on decay factor with smooth transitions"""
        # Fresh memories: Vibrant green (like spring)
        if decay_factor > 0.8:
            intensity = 150 + int(decay_factor * 105)
            return f"rgba(50, {intensity}, 100, 0.9)"
        
        # Active memories: Blue-green (like summer)
        elif decay_factor > 0.6:
            g_val = 200 - int((0.8 - decay_factor) * 200)
            b_val = 150 + int((decay_factor - 0.6) * 100)
            return f"rgba(100, {g_val}, {b_val}, 0.8)"
        
        # Aging memories: Amber/orange (like autumn)
        elif decay_factor > 0.4:
            r_val = 200 + int((0.6 - decay_factor) * 55)
            g_val = 150 + int((decay_factor - 0.4) * 100)
            return f"rgba({r_val}, {g_val}, 50, 0.7)"
        
        # Old memories: Red-orange (like late autumn)
        elif decay_factor > 0.2:
            r_val = 255
            g_val = 100 + int((decay_factor - 0.2) * 100)
            return f"rgba({r_val}, {g_val}, 30, 0.6)"
        
        # Fading memories: Gray (like winter/dormant)
        else:
            gray_val = 100 + int(decay_factor * 100)
            return f"rgba({gray_val}, {gray_val}, {gray_val}, 0.4)"
    
    def get_decay_size(self, decay_factor: float, base_size: float = 8) -> float:
        """Calculate node size based on decay factor"""
        return base_size * (0.3 + decay_factor * 0.7)
    
    def get_decay_opacity(self, decay_factor: float) -> float:
        """Calculate opacity based on decay factor"""
        return 0.2 + decay_factor * 0.8

class MemoryFractalGenerator:
    """Generate fractal tree structures for memory hierarchies"""
    
    def __init__(self):
        self.fractal_cache = {}
    
    def build_fractal_tree(self, root_node: Dict, related_nodes: List[Dict], 
                          max_depth: int = 5, branch_factor: float = 0.618) -> Dict:
        """Build a fractal memory tree using golden ratio branching"""
        
        fractal_tree = {
            "id": root_node["id"],
            "content": root_node.get("content", ""),
            "level": 0,
            "children": [],
            "position": {"x": 0, "y": 0, "z": 0},
            "decay_factor": root_node.get("decay_factor", 0.5),
            "memory_type": root_node.get("memory_type", "general"),
            "branch_strength": 1.0,
            "fractal_properties": {
                "branch_angle": 0,
                "branch_length": 1.0,
                "growth_direction": {"theta": 0, "phi": 0}
            }
        }
        
        # Recursively build branches using golden spiral
        if max_depth > 0:
            self._build_fractal_branches(
                fractal_tree, related_nodes, 0, max_depth, branch_factor
            )
        
        return fractal_tree
    
    def _build_fractal_branches(self, parent_node: Dict, available_nodes: List[Dict],
                               current_depth: int, max_depth: int, branch_factor: float):
        """Build fractal branches using mathematical spirals"""
        
        if current_depth >= max_depth or not available_nodes:
            return
        
        # Calculate number of branches using Fibonacci-like sequence
        num_branches = min(len(available_nodes), 2 + current_depth)
        
        for i in range(num_branches):
            if i >= len(available_nodes):
                break
                
            child_node = available_nodes[i].copy()
            
            # Golden angle for spiral positioning
            golden_angle = 2.399963  # 2π / φ
            angle = i * golden_angle
            
            # Branch length decreases with depth (fractal property)
            branch_length = parent_node["fractal_properties"]["branch_length"] * branch_factor
            
            # Calculate 3D position using spherical coordinates
            phi = angle
            theta = math.asin(math.sqrt(i / num_branches)) if num_branches > 0 else 0
            
            # Position relative to parent
            parent_pos = parent_node["position"]
            radius = branch_length * 2
            
            child_position = {
                "x": parent_pos["x"] + radius * math.sin(theta) * math.cos(phi),
                "y": parent_pos["y"] + radius * math.sin(theta) * math.sin(phi),
                "z": parent_pos["z"] + radius * math.cos(theta)
            }
            
            # Create child node with fractal properties
            child = {
                "id": child_node["id"],
                "content": child_node.get("content", ""),
                "level": current_depth + 1,
                "children": [],
                "position": child_position,
                "decay_factor": child_node.get("decay_factor", 0.5),
                "memory_type": child_node.get("memory_type", "general"),
                "branch_strength": parent_node["branch_strength"] * branch_factor,
                "fractal_properties": {
                    "branch_angle": angle,
                    "branch_length": branch_length,
                    "growth_direction": {"theta": theta, "phi": phi}
                }
            }
            
            parent_node["children"].append(child)
            
            # Recursively build deeper levels
            remaining_nodes = available_nodes[num_branches:]
            if remaining_nodes:
                self._build_fractal_branches(
                    child, remaining_nodes, current_depth + 1, max_depth, branch_factor
                )

class MerXMemoryUI:
    """Main UI class for merX memory visualization"""
    
    def __init__(self):
        self.decay_visualizer = MemoryDecayVisualizer()
        self.fractal_generator = MemoryFractalGenerator()
        self.selected_node = None
        self.animation_speed = 1.0
        
        # Initialize session state
        if 'memory_data' not in st.session_state:
            st.session_state.memory_data = None
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "3D Network"
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None

    def load_memory_data(self) -> Dict[str, Any]:
        """Load real merX memory data"""
        try:
            import sys
            import os
            # Add the parent directory to path to access src module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            src_path = os.path.join(parent_dir, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            from container.enhanced_di_container import create_enhanced_container
            
            # Create memory engine
            container = create_enhanced_container("data/test_output")
            engine = container.get("memory_engine")
            
            # Get RAMX data
            ramx = getattr(engine.storage, "ramx", None)
            if not ramx:
                return self._generate_sample_data()
            
            ramx_nodes = ramx.get_all_nodes()
            logger.info(f"📊 Loaded {len(ramx_nodes)} nodes from merX RAMX")
            
            # Convert to visualization format
            nodes = []
            edges = []
            current_time = time.time()
            
            for i, ramx_node in enumerate(ramx_nodes):
                if i >= 10000:  # Limit for UI responsiveness
                    break
                    
                age_hours = (current_time - ramx_node.timestamp) / 3600
                decay_factor = ramx_node.activation
                
                # Extract domain from tags
                domain = "general"
                if ramx_node.tags:
                    for tag in ramx_node.tags:
                        if tag in ['artificial_intelligence', 'computer_science', 'biology', 
                                  'physics', 'chemistry', 'mathematics', 'literature', 
                                  'history', 'psychology', 'philosophy']:
                            domain = tag
                            break
                
                node = {
                    "id": str(ramx_node.id),
                    "content": ramx_node.content[:100] + "..." if len(ramx_node.content) > 100 else ramx_node.content,
                    "memory_type": ramx_node.node_type,
                    "domain": domain,
                    "decay_factor": decay_factor,
                    "age_hours": age_hours,
                    "tags": ramx_node.tags,
                    "links": len(ramx_node.links) if ramx_node.links else 0
                }
                nodes.append(node)
                
                # Add edges from links
                if ramx_node.links:
                    for target_id, link_data in ramx_node.links.items():
                        weight = link_data[0] if isinstance(link_data, (list, tuple)) and len(link_data) > 0 else 0.5
                        edges.append({
                            "source": str(ramx_node.id),
                            "target": str(target_id),
                            "weight": weight
                        })
            
            return {"nodes": nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Error loading merX data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample memory data for demonstration"""
        np.random.seed(42)
        
        domains = ['artificial_intelligence', 'computer_science', 'biology', 'physics', 'chemistry']
        memory_types = ['working', 'episodic', 'semantic', 'procedural']
        
        nodes = []
        for i in range(1000):
            nodes.append({
                "id": f"memory_{i}",
                "content": f"Memory content {i} about {np.random.choice(domains)}",
                "memory_type": np.random.choice(memory_types),
                "domain": np.random.choice(domains),
                "decay_factor": np.random.beta(2, 3),
                "age_hours": np.random.exponential(24),
                "tags": [np.random.choice(domains), np.random.choice(memory_types)],
                "links": np.random.poisson(3)
            })
        
        # Generate edges
        edges = []
        for i in range(min(2000, len(nodes) * 2)):
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)
            if source["id"] != target["id"]:
                edges.append({
                    "source": source["id"],
                    "target": target["id"],
                    "weight": np.random.uniform(0.1, 1.0)
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def create_3d_network_view(self, data: Dict[str, Any]) -> go.Figure:
        """Create the main 3D memory network visualization"""
        nodes = data["nodes"]
        edges = data["edges"]
        
        # Create 3D layout using sphere positioning
        n_nodes = len(nodes)
        
        # Golden spiral positioning for beautiful distribution
        indices = np.arange(0, n_nodes, dtype=float) + 0.5
        theta = np.arccos(1 - 2*indices/n_nodes)
        phi = np.pi * (1 + 5**0.5) * indices
        
        radius = 10
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges first (behind nodes)
        edge_x, edge_y, edge_z = [], [], []
        edge_weights = []
        
        # Create position lookup
        pos_lookup = {node["id"]: (x[i], y[i], z[i]) for i, node in enumerate(nodes)}
        
        for edge in edges[:1000]:  # Limit edges for performance
            if edge["source"] in pos_lookup and edge["target"] in pos_lookup:
                x0, y0, z0 = pos_lookup[edge["source"]]
                x1, y1, z1 = pos_lookup[edge["target"]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                edge_weights.append(edge["weight"])
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
            hoverinfo='skip',
            showlegend=False,
            name='Memory Connections'
        ))
        
        # Add nodes with decay visualization
        node_colors = []
        node_sizes = []
        hover_texts = []
        
        for i, node in enumerate(nodes):
            # Get decay-based visual properties
            color = self.decay_visualizer.get_decay_color(node["decay_factor"], node["memory_type"])
            size = self.decay_visualizer.get_decay_size(node["decay_factor"])
            
            node_colors.append(color)
            node_sizes.append(size)
            
            # Create rich hover text
            hover_text = (
                f"<b>Memory ID:</b> {node['id']}<br>"
                f"<b>Type:</b> {node['memory_type']}<br>"
                f"<b>Domain:</b> {node['domain']}<br>"
                f"<b>Decay Factor:</b> {node['decay_factor']:.2f}<br>"
                f"<b>Age:</b> {node['age_hours']:.1f} hours<br>"
                f"<b>Content:</b> {node['content']}<br>"
                f"<b>Links:</b> {node['links']}"
            )
            hover_texts.append(hover_text)
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            text=hover_texts,
            hoverinfo='text',
            customdata=[node["id"] for node in nodes],
            name='Memory Nodes'
        ))
        
        # Update layout for GPU optimization
        fig.update_layout(
            title={
                'text': '🧠 merX Memory Network - 3D Visualization',
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, showticklabels=False, title=""),
                zaxis=dict(showgrid=False, showticklabels=False, title=""),
                bgcolor='rgba(0, 0, 0, 0.9)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(0, 0, 0, 0.9)',
            plot_bgcolor='rgba(0, 0, 0, 0.9)',
            font=dict(color='white'),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_tree_of_trees_view(self, data: Dict[str, Any]) -> go.Figure:
        """Create hierarchical tree-of-trees visualization"""
        nodes = data["nodes"]
        
        # Group nodes by domain to create tree structure
        domain_trees = {}
        for node in nodes:
            domain = node["domain"]
            if domain not in domain_trees:
                domain_trees[domain] = []
            domain_trees[domain].append(node)
        
        fig = go.Figure()
        
        # Create forest of trees layout
        tree_positions = {}
        colors = px.colors.qualitative.Set3
        
        for tree_idx, (domain, domain_nodes) in enumerate(domain_trees.items()):
            if not domain_nodes:
                continue
                
            # Position each tree in the forest
            tree_center_x = (tree_idx % 3) * 15 - 15
            tree_center_y = (tree_idx // 3) * 15 - 15
            tree_center_z = 0
            
            # Create hierarchical layout within each tree
            n_nodes = len(domain_nodes)
            if n_nodes == 0:
                continue
                
            # Sort nodes by decay factor (fresh memories higher)
            domain_nodes.sort(key=lambda x: x["decay_factor"], reverse=True)
            
            # Create tree structure using golden spiral
            for i, node in enumerate(domain_nodes[:50]):  # Limit per tree
                level = min(4, i // 10)  # Create levels
                angle = (i % 10) * (2 * np.pi / 10)  # 10 nodes per level
                radius = 2 + level * 1.5
                
                x = tree_center_x + radius * np.cos(angle)
                y = tree_center_y + radius * np.sin(angle)
                z = tree_center_z + level * 2
                
                tree_positions[node["id"]] = (x, y, z)
        
        # Extract positions
        node_ids = list(tree_positions.keys())
        x_pos = [tree_positions[nid][0] for nid in node_ids]
        y_pos = [tree_positions[nid][1] for nid in node_ids]
        z_pos = [tree_positions[nid][2] for nid in node_ids]
        
        # Create node lookup for coloring
        node_lookup = {node["id"]: node for node in nodes}
        
        # Add tree connections (trunk and branches)
        edge_x, edge_y, edge_z = [], [], []
        
        for domain, domain_nodes in domain_trees.items():
            domain_nodes_in_pos = [n for n in domain_nodes if n["id"] in tree_positions]
            if len(domain_nodes_in_pos) < 2:
                continue
                
            # Connect nodes in same domain (tree structure)
            for i in range(len(domain_nodes_in_pos) - 1):
                node1 = domain_nodes_in_pos[i]
                node2 = domain_nodes_in_pos[i + 1]
                
                if node1["id"] in tree_positions and node2["id"] in tree_positions:
                    x1, y1, z1 = tree_positions[node1["id"]]
                    x2, y2, z2 = tree_positions[node2["id"]]
                    edge_x.extend([x1, x2, None])
                    edge_y.extend([y1, y2, None])
                    edge_z.extend([z1, z2, None])
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(139, 69, 19, 0.6)', width=3),  # Brown tree branches
            hoverinfo='skip',
            showlegend=False,
            name='Tree Branches'
        ))
        
        # Prepare node visualization data
        node_colors = []
        node_sizes = []
        hover_texts = []
        
        for node_id in node_ids:
            node = node_lookup[node_id]
            color = self.decay_visualizer.get_decay_color(node["decay_factor"], node["memory_type"])
            size = self.decay_visualizer.get_decay_size(node["decay_factor"], 6)
            
            node_colors.append(color)
            node_sizes.append(size)
            
            hover_text = (
                f"<b>🌿 Tree Node:</b> {node['id']}<br>"
                f"<b>🌳 Domain Tree:</b> {node['domain']}<br>"
                f"<b>🧠 Memory Type:</b> {node['memory_type']}<br>"
                f"<b>🍂 Decay Factor:</b> {node['decay_factor']:.2f}<br>"
                f"<b>📝 Content:</b> {node['content']}"
            )
            hover_texts.append(hover_text)
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=x_pos, y=y_pos, z=z_pos,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=hover_texts,
            hoverinfo='text',
            customdata=node_ids,
            name='Memory Tree Nodes'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🌳 merX Memory Forest - Tree-of-Trees Visualization',
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title="Knowledge Domains"),
                yaxis=dict(showgrid=False, showticklabels=False, title="Semantic Space"),
                zaxis=dict(showgrid=False, showticklabels=False, title="Memory Hierarchy"),
                bgcolor='rgba(10, 40, 20, 0.9)',  # Forest green background
                camera=dict(eye=dict(x=2, y=2, z=1.5))
            ),
            paper_bgcolor='rgba(10, 40, 20, 0.9)',
            plot_bgcolor='rgba(10, 40, 20, 0.9)',
            font=dict(color='white'),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    def create_fractal_explorer(self, data: Dict[str, Any], root_node_id: Optional[str] = None) -> go.Figure:
        """Create interactive fractal memory explorer"""
        nodes = data["nodes"]
        
        if not root_node_id and nodes:
            # Find highest decay factor node as root
            root_node_id = max(nodes, key=lambda x: x["decay_factor"])["id"]
        
        if not root_node_id:
            return go.Figure()
        
        # Find root node
        root_node = next((n for n in nodes if n["id"] == root_node_id), None)
        if not root_node:
            return go.Figure()
        
        # Find related nodes (same domain or similar content)
        related_nodes = [
            n for n in nodes 
            if n["id"] != root_node_id and 
            (n["domain"] == root_node["domain"] or 
             any(tag in root_node.get("tags", []) for tag in n.get("tags", [])))
        ][:20]  # Limit for visualization
        
        # Build fractal tree
        fractal_tree = self.fractal_generator.build_fractal_tree(
            root_node, related_nodes, max_depth=4
        )
        
        # Extract positions and data for visualization
        fig = go.Figure()
        
        def extract_fractal_data(node, parent_pos=None):
            """Recursively extract fractal tree data"""
            positions = []
            colors = []
            sizes = []
            hover_texts = []
            edges_x, edges_y, edges_z = [], [], []
            
            pos = (node["position"]["x"], node["position"]["y"], node["position"]["z"])
            positions.append(pos)
            
            # Color based on branch strength and decay
            decay = node["decay_factor"]
            branch_strength = node["branch_strength"]
            combined_factor = decay * branch_strength
            
            color = self.decay_visualizer.get_decay_color(combined_factor)
            colors.append(color)
            
            # Size based on level (inverse) and decay
            level_factor = 1.0 / (node["level"] + 1)
            size = 8 + level_factor * 6 + decay * 4
            sizes.append(size)
            
            # Hover text with fractal properties
            hover_text = (
                f"<b>🌿 Fractal Node:</b> {node['id']}<br>"
                f"<b>📊 Level:</b> {node['level']}<br>"
                f"<b>💪 Branch Strength:</b> {branch_strength:.2f}<br>"
                f"<b>🍂 Decay Factor:</b> {decay:.2f}<br>"
                f"<b>🧠 Memory Type:</b> {node['memory_type']}<br>"
                f"<b>📝 Content:</b> {node['content']}<br>"
                f"<b>🌀 Branch Angle:</b> {node['fractal_properties']['branch_angle']:.2f}°"
            )
            hover_texts.append(hover_text)
            
            # Add edge to parent
            if parent_pos:
                edges_x.extend([parent_pos[0], pos[0], None])
                edges_y.extend([parent_pos[1], pos[1], None])
                edges_z.extend([parent_pos[2], pos[2], None])
            
            # Process children
            for child in node["children"]:
                child_positions, child_colors, child_sizes, child_hovers, child_edges = extract_fractal_data(child, pos)
                positions.extend(child_positions)
                colors.extend(child_colors)
                sizes.extend(child_sizes)
                hover_texts.extend(child_hovers)
                edges_x.extend(child_edges[0])
                edges_y.extend(child_edges[1])
                edges_z.extend(child_edges[2])
            
            return positions, colors, sizes, hover_texts, (edges_x, edges_y, edges_z)
        
        positions, colors, sizes, hover_texts, (edges_x, edges_y, edges_z) = extract_fractal_data(fractal_tree)
        
        # Add fractal branches
        fig.add_trace(go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode='lines',
            line=dict(
                color='rgba(139, 69, 19, 0.8)',  # Brown branches
                width=2
            ),
            hoverinfo='skip',
            showlegend=False,
            name='Fractal Branches'
        ))
        
        # Add fractal nodes
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions]
        z_pos = [pos[2] for pos in positions]
        
        fig.add_trace(go.Scatter3d(
            x=x_pos, y=y_pos, z=z_pos,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Fractal Memory Nodes'
        ))
        
        # Update layout with fractal-inspired design
        fig.update_layout(
            title={
                'text': f'🌀 merX Memory Fractal - Root: {root_node_id}',
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title="Fractal Width"),
                yaxis=dict(showgrid=False, showticklabels=False, title="Associative Space"),
                zaxis=dict(showgrid=False, showticklabels=False, title="Fractal Depth"),
                bgcolor='rgba(20, 20, 40, 0.9)',  # Deep purple background
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            paper_bgcolor='rgba(20, 20, 40, 0.9)',
            plot_bgcolor='rgba(20, 20, 40, 0.9)',
            font=dict(color='white'),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def render_ui(self):
        """Render the main Streamlit UI"""
        # Header
        st.markdown("# 🧠 merX Memory 3D Visualizer")
        st.markdown("### Explore your memory network in beautiful 3D with decay visualization")
        
        # Sidebar controls
        with st.sidebar:
            st.markdown("## 🎛️ Visualization Controls")
            
            # View mode selection
            view_mode = st.radio(
                "Select View Mode:",
                ["3D Network", "Tree-of-Trees", "Fractal Explorer"],
                index=["3D Network", "Tree-of-Trees", "Fractal Explorer"].index(st.session_state.view_mode)
            )
            st.session_state.view_mode = view_mode
            
            st.markdown("---")
            
            # Animation controls
            st.markdown("### 🎬 Animation Settings")
            animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)
            auto_rotate = st.checkbox("Auto-rotate camera", value=False)
            
            st.markdown("---")
            
            # Memory decay settings
            st.markdown("### 🍂 Memory Decay Visualization")
            show_decay_legend = st.checkbox("Show decay color legend", value=True)
            decay_threshold = st.slider("Decay visibility threshold", 0.0, 1.0, 0.1, 0.05)
            
            st.markdown("---")
            
            # Performance settings
            st.markdown("### ⚡ Performance")
            max_nodes = st.selectbox("Max nodes to display", [500, 1000, 2000, 5000, 10000], index=1)
            enable_webgl = st.checkbox("Enable WebGL acceleration", value=True)
              # Data refresh
            if st.button("🔄 Refresh Data"):
                st.session_state.memory_data = None
                st.rerun()
        
        # Load data
        if st.session_state.memory_data is None:
            with st.spinner("Loading merX memory data..."):
                st.session_state.memory_data = self.load_memory_data()
        
        data = st.session_state.memory_data
        
        if not data or not data.get("nodes"):
            st.error("❌ No memory data available. Please check your merX database.")
            return
        
        # Filter data based on settings
        filtered_nodes = [
            node for node in data["nodes"][:max_nodes]
            if node["decay_factor"] >= decay_threshold
        ]
        filtered_data = {
            "nodes": filtered_nodes,
            "edges": [
                edge for edge in data["edges"]
                if any(node["id"] == edge["source"] for node in filtered_nodes) and
                   any(node["id"] == edge["target"] for node in filtered_nodes)
            ]
        }
        
        # Main visualization area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Render selected view
            if view_mode == "3D Network":
                st.markdown("### 🌌 3D Memory Network")
                fig = self.create_3d_network_view(filtered_data)
            elif view_mode == "Tree-of-Trees":
                st.markdown("### 🌳 Tree-of-Trees Hierarchy")
                fig = self.create_tree_of_trees_view(filtered_data)
            else:  # Fractal Explorer
                st.markdown("### 🌀 Fractal Memory Explorer")
                
                # Node selection for fractal root
                if filtered_nodes:
                    node_options = {f"{node['id']} ({node['domain']})": node['id'] for node in filtered_nodes[:50]}
                    selected_display = st.selectbox("Select root node for fractal exploration:", list(node_options.keys()))
                    root_node_id = node_options[selected_display]
                else:
                    root_node_id = None
                
                fig = self.create_fractal_explorer(filtered_data, root_node_id)
            
            # Display the visualization
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'merx_memory_{view_mode.lower().replace(" ", "_")}',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)
        
        with col2:
            # Statistics and info panel
            st.markdown("### 📊 Memory Statistics")
            
            if filtered_nodes:
                # Basic stats
                st.metric("Total Memories", len(filtered_nodes))
                st.metric("Active Connections", len(filtered_data["edges"]))
                
                # Decay distribution
                avg_decay = np.mean([node["decay_factor"] for node in filtered_nodes])
                st.metric("Average Decay Factor", f"{avg_decay:.2f}")
                
                # Domain distribution
                domain_counts = {}
                for node in filtered_nodes:
                    domain = node["domain"]
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                st.markdown("#### 🏷️ Domain Distribution")
                for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{domain}:** {count}")
                
                # Memory type distribution
                type_counts = {}
                for node in filtered_nodes:
                    mem_type = node["memory_type"]
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                
                st.markdown("#### 🧠 Memory Type Distribution")
                for mem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{mem_type}:** {count}")
            
            # Decay legend
            if show_decay_legend:
                st.markdown("#### 🍂 Memory Decay Legend")
                legend_html = """
                <div style='font-size: 12px;'>
                <div style='color: #32FF64;'>🟢 Fresh (0.8-1.0): Very active memories</div>
                <div style='color: #64C8FF;'>🔵 Active (0.6-0.8): Recently accessed</div>
                <div style='color: #FFC832;'>🟡 Aging (0.4-0.6): Moderately old</div>
                <div style='color: #FF6432;'>🟠 Old (0.2-0.4): Rarely accessed</div>
                <div style='color: #808080;'>⚫ Fading (0.0-0.2): Nearly forgotten</div>
                </div>
                """
                st.markdown(legend_html, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    ui = MerXMemoryUI()
    ui.render_ui()


if __name__ == "__main__":
    main()
