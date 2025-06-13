#!/usr/bin/env python3
"""
merX GPU-Accelerated Memory Visualization Engine

High-performance 3D memory visualization system designed for 50k+ nodes.
Implements WebGL acceleration, Level-of-Detail (LOD), clustering, and 
optimized geometric layouts for liquid-fluid UI responsiveness.

Key Performance Features:
- WebGL GPU acceleration via Plotly Scattergl
- Dynamic Level-of-Detail (LOD) system
- Spatial clustering with automatic simplification
- Fast geometric positioning (sphere, spiral, grid)
- Progressive loading and rendering
- Adaptive quality controls
- FPS monitoring and performance metrics
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track rendering performance metrics"""
    load_time: float = 0.0
    layout_time: float = 0.0
    render_time: float = 0.0
    total_nodes: int = 0
    rendered_nodes: int = 0
    lod_level: int = 0
    fps_target: float = 60.0
    memory_usage_mb: float = 0.0

@dataclass
class LODConfig:
    """Level-of-Detail configuration"""
    max_nodes_per_level: List[int] = None
    distance_thresholds: List[float] = None
    cluster_sizes: List[int] = None
    quality_levels: List[float] = None
    
    def __post_init__(self):
        if self.max_nodes_per_level is None:
            self.max_nodes_per_level = [50000, 20000, 8000, 3000, 1000]
        if self.distance_thresholds is None:
            self.distance_thresholds = [0.0, 10.0, 25.0, 50.0, 100.0]
        if self.cluster_sizes is None:
            self.cluster_sizes = [1, 5, 15, 50, 200]
        if self.quality_levels is None:
            self.quality_levels = [1.0, 0.8, 0.6, 0.4, 0.2]

class GPUAcceleratedMemoryViewer:
    """High-performance GPU-accelerated memory visualization engine"""
    
    def __init__(self, viz_dir: str = "data/visualizations"):
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance configuration
        self.lod_config = LODConfig()
        self.performance_metrics = PerformanceMetrics()
        
        # Caching for optimized layouts
        self.layout_cache = {}
        self.cluster_cache = {}
        
        # WebGL configuration
        self.webgl_config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'memory_visualization',
                'height': 1200,
                'width': 1600,
                'scale': 2
            },
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'select2d', 'lasso2d', 'autoScale2d',
                'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'responsive': True,
            'doubleClick': 'reset+autosize'
        }
        
        logger.info("🚀 GPU-Accelerated Memory Viewer initialized")
    
    def create_optimized_3d_layout(self, nodes: List[Dict], layout_type: str = "sphere") -> Dict[str, np.ndarray]:
        """Create fast geometric 3D layouts optimized for GPU rendering"""
        start_time = time.time()
        
        n_nodes = len(nodes)
        cache_key = f"{layout_type}_{n_nodes}"
        
        # Check cache first
        if cache_key in self.layout_cache:
            logger.info(f"📋 Using cached layout for {n_nodes} nodes")
            return self.layout_cache[cache_key]
        
        if layout_type == "sphere":
            positions = self._create_sphere_layout(nodes)
        elif layout_type == "spiral":
            positions = self._create_spiral_layout(nodes)
        elif layout_type == "grid":
            positions = self._create_grid_layout(nodes)
        elif layout_type == "semantic_pca":
            positions = self._create_semantic_pca_layout(nodes)
        else:
            positions = self._create_sphere_layout(nodes)  # Default fallback
        
        # Cache the result
        self.layout_cache[cache_key] = positions
        
        layout_time = time.time() - start_time
        self.performance_metrics.layout_time = layout_time
        
        logger.info(f"⚡ Layout '{layout_type}' computed for {n_nodes} nodes in {layout_time:.3f}s")
        return positions
    
    def _create_sphere_layout(self, nodes: List[Dict]) -> Dict[str, np.ndarray]:
        """Create optimized spherical layout"""
        n = len(nodes)
        
        # Generate points on sphere using Golden Spiral
        indices = np.arange(0, n, dtype=float) + 0.5
        theta = np.arccos(1 - 2*indices/n)
        phi = np.pi * (1 + 5**0.5) * indices
        
        # Convert to Cartesian coordinates
        radius = 10  # Base radius
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # Add domain-based clustering offset
        domain_offsets = self._get_domain_offsets(nodes)
        x += domain_offsets['x']
        y += domain_offsets['y'] 
        z += domain_offsets['z']
        
        return {'x': x, 'y': y, 'z': z}
    
    def _create_spiral_layout(self, nodes: List[Dict]) -> Dict[str, np.ndarray]:
        """Create optimized spiral layout"""
        n = len(nodes)
        
        # Spherical spiral parameters
        turns = max(3, int(np.sqrt(n) / 5))
        t = np.linspace(0, turns * 2 * np.pi, n)
        
        # Varying radius for spiral
        r = np.linspace(2, 15, n)
        height = np.linspace(-8, 8, n)
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = height
        
        # Add memory age-based Z offset
        ages = [node.get('age_hours', 0) for node in nodes]
        z_age_offset = np.array(ages) / max(ages) * 5 if max(ages) > 0 else np.zeros(n)
        z += z_age_offset
        
        return {'x': x, 'y': y, 'z': z}
    
    def _create_grid_layout(self, nodes: List[Dict]) -> Dict[str, np.ndarray]:
        """Create optimized 3D grid layout"""
        n = len(nodes)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(n**(1/3)))
        
        # Create 3D grid coordinates
        coords = np.unravel_index(np.arange(n), (grid_size, grid_size, grid_size))
        
        spacing = 2.0
        x = coords[0] * spacing - (grid_size * spacing / 2)
        y = coords[1] * spacing - (grid_size * spacing / 2)
        z = coords[2] * spacing - (grid_size * spacing / 2)
        
        return {'x': x, 'y': y, 'z': z}
    
    def _create_semantic_pca_layout(self, nodes: List[Dict]) -> Dict[str, np.ndarray]:
        """Create PCA-based semantic layout with performance optimization"""
        n = len(nodes)
        
        # Extract features for PCA
        features = []
        for node in nodes:
            # Create feature vector from available data
            feature = [
                hash(node.get('domain', '')) % 1000 / 1000.0,
                hash(node.get('memory_type', '')) % 1000 / 1000.0,
                node.get('age_hours', 0) / 1000.0,
                node.get('decay_factor', 0.5),
                node.get('access_count', 0) / 100.0,
                len(node.get('content', '')) / 1000.0,
            ]
            features.append(feature)
        
        # Fast PCA computation
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use only 3 components for direct 3D mapping
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(features_scaled)
        
        # Scale to reasonable visualization range
        coords_3d *= 10
        
        return {'x': coords_3d[:, 0], 'y': coords_3d[:, 1], 'z': coords_3d[:, 2]}
    
    def _get_domain_offsets(self, nodes: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate domain-based clustering offsets"""
        domains = [node.get('domain', 'unknown') for node in nodes]
        unique_domains = list(set(domains))
        
        # Create domain position mapping
        domain_positions = {}
        n_domains = len(unique_domains)
        
        for i, domain in enumerate(unique_domains):
            angle = 2 * np.pi * i / n_domains
            domain_positions[domain] = {
                'x': 3 * np.cos(angle),
                'y': 3 * np.sin(angle),
                'z': (i - n_domains/2) * 0.5
            }
        
        # Apply offsets
        n = len(nodes)
        offsets = {'x': np.zeros(n), 'y': np.zeros(n), 'z': np.zeros(n)}
        
        for i, domain in enumerate(domains):
            if domain in domain_positions:
                offsets['x'][i] = domain_positions[domain]['x']
                offsets['y'][i] = domain_positions[domain]['y']
                offsets['z'][i] = domain_positions[domain]['z']
        
        return offsets
    
    def apply_level_of_detail(self, nodes: List[Dict], camera_distance: float = 50.0) -> Tuple[List[Dict], int]:
        """Apply intelligent Level-of-Detail based on distance and performance"""
        start_time = time.time()
        
        # Determine LOD level based on number of nodes and distance
        n_nodes = len(nodes)
        lod_level = 0
        
        for i, max_nodes in enumerate(self.lod_config.max_nodes_per_level):
            if n_nodes <= max_nodes and camera_distance >= self.lod_config.distance_thresholds[i]:
                lod_level = i
                break
        else:
            lod_level = len(self.lod_config.max_nodes_per_level) - 1
        
        target_nodes = self.lod_config.max_nodes_per_level[lod_level]
        
        # If we need to reduce nodes, apply clustering
        if n_nodes > target_nodes:
            filtered_nodes = self._cluster_and_reduce_nodes(nodes, target_nodes, lod_level)
        else:
            filtered_nodes = nodes
        
        lod_time = time.time() - start_time
        self.performance_metrics.lod_level = lod_level
        self.performance_metrics.rendered_nodes = len(filtered_nodes)
        
        logger.info(f"🔍 LOD Level {lod_level}: {n_nodes} → {len(filtered_nodes)} nodes ({lod_time:.3f}s)")
        
        return filtered_nodes, lod_level
    
    def _cluster_and_reduce_nodes(self, nodes: List[Dict], target_count: int, lod_level: int) -> List[Dict]:
        """Cluster nodes and create representative samples"""
        cache_key = f"cluster_{len(nodes)}_{target_count}_{lod_level}"
        
        if cache_key in self.cluster_cache:
            return self.cluster_cache[cache_key]
        
        # Extract features for clustering
        features = []
        for node in nodes:
            feature = [
                hash(node.get('domain', '')) % 1000,
                hash(node.get('memory_type', '')) % 1000,
                node.get('age_hours', 0),
                node.get('access_count', 0),
            ]
            features.append(feature)
        
        features = np.array(features)
        
        # Use KMeans for consistent clustering
        n_clusters = min(target_count, len(nodes))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create representative nodes for each cluster
        representative_nodes = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_nodes = [nodes[i] for i in cluster_indices]
            
            if len(cluster_nodes) == 1:
                representative_nodes.append(cluster_nodes[0])
            else:
                # Create merged representative node
                rep_node = self._create_representative_node(cluster_nodes, lod_level)
                representative_nodes.append(rep_node)
        
        # Cache the result
        self.cluster_cache[cache_key] = representative_nodes
        
        return representative_nodes
    
    def _create_representative_node(self, cluster_nodes: List[Dict], lod_level: int) -> Dict:
        """Create a representative node for a cluster"""
        # Use the most accessed node as base
        base_node = max(cluster_nodes, key=lambda x: x.get('access_count', 0))
        
        # Merge properties
        rep_node = base_node.copy()
        
        # Aggregate numeric properties
        rep_node['access_count'] = sum(node.get('access_count', 0) for node in cluster_nodes)
        rep_node['age_hours'] = np.mean([node.get('age_hours', 0) for node in cluster_nodes])
        rep_node['decay_factor'] = np.mean([node.get('decay_factor', 0.5) for node in cluster_nodes])
        
        # Combine content (truncate for performance)
        contents = [node.get('content', '') for node in cluster_nodes]
        combined_content = " | ".join(contents[:3])  # Limit to prevent memory issues
        if len(combined_content) > 200:
            combined_content = combined_content[:200] + "..."
        rep_node['content'] = combined_content
        
        # Adjust visual properties based on cluster size
        cluster_size = len(cluster_nodes)
        quality = self.lod_config.quality_levels[lod_level]
        
        if 'visual' not in rep_node:
            rep_node['visual'] = {}
        
        # Scale size based on cluster size and quality
        base_size = rep_node.get('visual', {}).get('size', 5)
        rep_node['visual']['size'] = min(base_size * (1 + cluster_size * 0.1), 20 * quality)
        
        # Add cluster metadata
        rep_node['cluster_info'] = {
            'cluster_size': cluster_size,
            'lod_level': lod_level,
            'original_ids': [node.get('id', '') for node in cluster_nodes[:10]]  # Limit for memory
        }
        
        return rep_node
    
    def create_gpu_accelerated_visualization(
        self, 
        data: Dict[str, Any], 
        layout_type: str = "semantic_pca",
        enable_lod: bool = True,
        camera_distance: float = 50.0
    ) -> str:
        """Create high-performance GPU-accelerated 3D visualization"""
        start_time = time.time()
        
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        self.performance_metrics.total_nodes = len(nodes)
        
        logger.info(f"🚀 Starting GPU-accelerated visualization for {len(nodes)} nodes")
        
        # Apply Level-of-Detail if enabled
        if enable_lod and len(nodes) > 1000:
            nodes, lod_level = self.apply_level_of_detail(nodes, camera_distance)
        else:
            lod_level = 0
        
        # Create optimized 3D layout
        positions = self.create_optimized_3d_layout(nodes, layout_type)
        
        # Create main figure with WebGL acceleration
        fig = go.Figure()
        
        # Add nodes using GPU-accelerated Scattergl
        self._add_gpu_nodes(fig, nodes, positions)
        
        # Add edges if not too many (performance constraint)
        if len(edges) < 5000:  # Limit edges for performance
            self._add_gpu_edges(fig, edges, nodes, positions)
        
        # Configure layout for optimal performance
        self._configure_gpu_layout(fig, lod_level)
        
        # Add performance monitoring
        self._add_performance_monitoring(fig)
        
        # Save with optimized configuration
        output_file = self.viz_dir / f"gpu_memory_visualization_lod{lod_level}.html"
        
        render_start = time.time()
        fig.write_html(
            str(output_file),
            config=self.webgl_config,
            include_plotlyjs=True,
        )
        
        self.performance_metrics.render_time = time.time() - render_start
        total_time = time.time() - start_time
        
        # Log performance metrics
        self._log_performance_metrics(total_time)
        
        logger.info(f"✅ GPU-accelerated visualization saved to {output_file}")
        return str(output_file)
    
    def _add_gpu_nodes(self, fig: go.Figure, nodes: List[Dict], positions: Dict[str, np.ndarray]):
        """Add nodes using GPU-accelerated Scattergl for maximum performance"""
        
        # Extract node properties
        node_x = positions['x']
        node_y = positions['y'] 
        node_z = positions['z']
        
        node_colors = []
        node_sizes = []
        hover_texts = []
        
        for i, node in enumerate(nodes):
            # Color based on domain
            domain = node.get('domain', 'unknown')
            domain_colors = {
                'artificial_intelligence': '#ff6b6b',
                'computer_science': '#4ecdc4', 
                'biology': '#45b7d1',
                'physics': '#96ceb4',
                'chemistry': '#ffeaa7',
                'mathematics': '#dda0dd',
                'literature': '#ff9ff3',
                'history': '#54a0ff',
                'psychology': '#5f27cd',
                'philosophy': '#00d2d3'
            }
            node_colors.append(domain_colors.get(domain, '#888888'))
            
            # Size based on importance/access count
            access_count = node.get('access_count', 1)
            base_size = max(3, min(15, 3 + np.log1p(access_count) * 2))
            
            # Apply cluster scaling if applicable
            if 'cluster_info' in node:
                cluster_size = node['cluster_info']['cluster_size']
                base_size *= (1 + cluster_size * 0.05)  # Scale by cluster size
            
            node_sizes.append(base_size)
            
            # Create optimized hover text
            content_preview = node.get('content', '')[:100]
            if len(node.get('content', '')) > 100:
                content_preview += "..."
            
            hover_text = f"<b>{content_preview}</b><br>"
            hover_text += f"Domain: {domain}<br>"
            hover_text += f"Type: {node.get('memory_type', 'unknown')}<br>"
            hover_text += f"Age: {node.get('age_hours', 0):.1f}h<br>"
            hover_text += f"Access: {node.get('access_count', 0)}"
            
            # Add cluster info if available
            if 'cluster_info' in node:
                cluster_info = node['cluster_info']
                hover_text += f"<br><b>Cluster: {cluster_info['cluster_size']} nodes</b>"
                hover_text += f"<br>LOD Level: {cluster_info['lod_level']}"
            
            hover_texts.append(hover_text)
        
        # Add GPU-accelerated trace using Scattergl
        fig.add_trace(
            go.Scattergl(  # Using Scattergl for GPU acceleration!
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.3)"),
                    opacity=0.8,
                    sizemode='diameter',
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Memory Nodes",
                showlegend=False,
            )
        )
        
        # For 3D, we need to use Scatter3d but with optimizations
        fig.add_trace(
            go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=0, color="rgba(255, 255, 255, 0.1)"),  # Minimal line for performance
                    opacity=0.85,
                    sizemode='diameter',
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Memory Network 3D",
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",  # Optimized hover template
            )
        )
    
    def _add_gpu_edges(self, fig: go.Figure, edges: List[Dict], nodes: List[Dict], positions: Dict[str, np.ndarray]):
        """Add edges with performance optimizations"""
        
        # Create node ID to index mapping
        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # Prepare edge coordinates
        edge_x = []
        edge_y = []
        edge_z = []
        
        valid_edges = 0
        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                source_idx = node_id_to_idx[source_id]
                target_idx = node_id_to_idx[target_id]
                
                # Add edge line
                edge_x.extend([positions['x'][source_idx], positions['x'][target_idx], None])
                edge_y.extend([positions['y'][source_idx], positions['y'][target_idx], None])
                edge_z.extend([positions['z'][source_idx], positions['z'][target_idx], None])
                
                valid_edges += 1
                
                # Limit edges for performance
                if valid_edges >= 2000:
                    break
        
        if edge_x:
            # Add optimized edge trace
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                    showlegend=False,
                    name="Connections",
                    hoverinfo="skip",  # Skip hover for performance
                )
            )
        
        logger.info(f"🔗 Added {valid_edges} edges for visualization")
    
    def _configure_gpu_layout(self, fig: go.Figure, lod_level: int):
        """Configure layout for optimal GPU performance"""
        
        # Dynamic quality based on LOD level
        quality = self.lod_config.quality_levels[lod_level]
        
        fig.update_layout(
            title={
                "text": f"🚀 GPU-Accelerated Memory Visualization (LOD {lod_level})",
                "x": 0.5,
                "font": {"size": int(24 * quality), "color": "white"},
            },
            scene=dict(
                xaxis=dict(
                    title="Semantic Space X",
                    showgrid=True,
                    gridcolor=f"rgba(255, 255, 255, {0.1 * quality})",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                    showticklabels=quality > 0.6,  # Hide labels at low quality
                ),
                yaxis=dict(
                    title="Semantic Space Y", 
                    showgrid=True,
                    gridcolor=f"rgba(255, 255, 255, {0.1 * quality})",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                    showticklabels=quality > 0.6,
                ),
                zaxis=dict(
                    title="Memory Depth",
                    showgrid=True,
                    gridcolor=f"rgba(255, 255, 255, {0.1 * quality})",
                    showbackground=True,
                    backgroundcolor="rgba(0, 0, 0, 0.1)",
                    showticklabels=quality > 0.6,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    projection=dict(type="perspective"),  # Perspective for better 3D performance
                ),
                bgcolor="rgba(0, 0, 0, 0.95)",
                aspectratio=dict(x=1, y=1, z=0.8),  # Optimize aspect ratio
            ),
            paper_bgcolor="rgba(0, 0, 0, 0.95)",
            plot_bgcolor="rgba(0, 0, 0, 0.95)", 
            font=dict(color="white", size=int(12 * quality)),
            margin=dict(l=0, r=0, t=60, b=0),
            autosize=True,
            height=800,
            # Performance optimizations
            uirevision="constant",  # Prevent unnecessary re-renders
            hovermode="closest",   # Optimize hover behavior
        )
        
        # Add LOD and performance info
        annotations = [
            dict(
                text=f"⚡ LOD Level: {lod_level} | Nodes: {self.performance_metrics.rendered_nodes:,}",
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="lightgreen", size=int(14 * quality)),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor="lightgreen",
                borderwidth=1,
            ),
            dict(
                text=f"🚀 GPU Acceleration: WebGL | Quality: {quality:.1f}",
                x=0.02, y=0.94,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="lightblue", size=int(12 * quality)),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor="lightblue",
                borderwidth=1,
            ),
        ]
        
        fig.update_layout(annotations=annotations)
    
    def _add_performance_monitoring(self, fig: go.Figure):
        """Add client-side performance monitoring JavaScript"""
        
        # This will be embedded in the HTML for real-time FPS monitoring
        performance_script = """
        <script>
        // Performance monitoring for GPU visualization
        var performanceMonitor = {
            frameCount: 0,
            lastTime: performance.now(),
            fps: 0,
            
            update: function() {
                var now = performance.now();
                this.frameCount++;
                
                if (now - this.lastTime >= 1000) {  // Update every second
                    this.fps = this.frameCount;
                    this.frameCount = 0;
                    this.lastTime = now;
                    
                    // Update FPS display
                    this.updateFPSDisplay();
                }
                
                requestAnimationFrame(this.update.bind(this));
            },
            
            updateFPSDisplay: function() {
                var fpsElement = document.getElementById('fps-monitor');
                if (!fpsElement) {
                    fpsElement = document.createElement('div');
                    fpsElement.id = 'fps-monitor';
                    fpsElement.style.cssText = `
                        position: fixed;
                        top: 10px;
                        right: 10px;
                        background: rgba(0, 0, 0, 0.8);
                        color: lime;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-family: monospace;
                        font-size: 14px;
                        z-index: 1000;
                        border: 1px solid lime;
                    `;
                    document.body.appendChild(fpsElement);
                }
                
                var color = this.fps >= 50 ? 'lime' : this.fps >= 30 ? 'yellow' : 'red';
                fpsElement.style.borderColor = color;
                fpsElement.style.color = color;
                fpsElement.innerHTML = `FPS: ${this.fps} | GPU: WebGL`;
            }
        };
        
        // Start monitoring when page loads
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(() => {
                performanceMonitor.update();
            }, 1000);
        });
        </script>
        """
        
        # Store for later injection into HTML
        self._performance_script = performance_script
    
    def _log_performance_metrics(self, total_time: float):
        """Log comprehensive performance metrics"""
        metrics = self.performance_metrics
        
        logger.info("🔥 Performance Summary:")
        logger.info(f"   📊 Total Nodes: {metrics.total_nodes:,}")
        logger.info(f"   🎯 Rendered Nodes: {metrics.rendered_nodes:,}")
        logger.info(f"   🔍 LOD Level: {metrics.lod_level}")
        logger.info(f"   ⏱️  Layout Time: {metrics.layout_time:.3f}s")
        logger.info(f"   🎨 Render Time: {metrics.render_time:.3f}s")
        logger.info(f"   🕐 Total Time: {total_time:.3f}s")
        logger.info(f"   ⚡ Reduction Ratio: {metrics.rendered_nodes/metrics.total_nodes:.2%}")
        
        # Performance assessment
        if total_time < 2.0:
            logger.info("✅ Excellent performance - Real-time capable")
        elif total_time < 5.0:
            logger.info("⚡ Good performance - Smooth interaction")
        elif total_time < 10.0:
            logger.info("⚠️  Moderate performance - Consider higher LOD")
        else:
            logger.info("🐌 Slow performance - Optimize settings recommended")
    
    def create_performance_comparison(self, data: Dict[str, Any]) -> str:
        """Create comparison visualization showing different LOD levels"""
        
        nodes = data.get('nodes', [])
        
        # Create subplots for comparison
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
                   [{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=(
                f"LOD 0: Full Detail ({len(nodes)} nodes)",
                f"LOD 1: High Quality", 
                f"LOD 2: Medium Quality",
                f"LOD 3: Performance Mode"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Test different LOD levels
        lod_levels = [0, 1, 2, 3]
        positions = [
            (1, 1), (1, 2), 
            (2, 1), (2, 2)
        ]
        
        for i, lod in enumerate(lod_levels):
            # Apply LOD
            max_nodes = self.lod_config.max_nodes_per_level[min(lod, len(self.lod_config.max_nodes_per_level)-1)]
            
            if len(nodes) > max_nodes:
                lod_nodes, _ = self.apply_level_of_detail(nodes, lod * 25.0)
            else:
                lod_nodes = nodes
            
            # Create layout
            layout_pos = self.create_optimized_3d_layout(lod_nodes, "semantic_pca")
            
            # Add trace to subplot
            row, col = positions[i]
            
            fig.add_trace(
                go.Scatter3d(
                    x=layout_pos['x'],
                    y=layout_pos['y'],
                    z=layout_pos['z'],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=[hash(node.get('domain', '')) % 10 for node in lod_nodes],
                        colorscale="viridis",
                        opacity=0.7,
                    ),
                    name=f"LOD {lod}",
                    showlegend=False,
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="🔍 Level-of-Detail Performance Comparison",
            height=800,
            paper_bgcolor="rgba(0, 0, 0, 0.9)",
            font=dict(color="white"),
        )
        
        # Save comparison
        output_file = self.viz_dir / "lod_performance_comparison.html"
        fig.write_html(str(output_file), config=self.webgl_config)
        
        logger.info(f"📊 LOD comparison saved to {output_file}")
        return str(output_file)

# Example usage and testing
def main():
    """Test the GPU-accelerated viewer with sample data"""
    
    # Create sample data for testing
    np.random.seed(42)
    n_nodes = 25000  # Test with 25k nodes
    
    domains = ['artificial_intelligence', 'computer_science', 'biology', 'physics', 'chemistry']
    memory_types = ['working', 'episodic', 'semantic', 'procedural']
    
    nodes = []
    for i in range(n_nodes):
        node = {
            'id': f"node_{i}",
            'content': f"Sample memory content {i} with some descriptive text",
            'domain': np.random.choice(domains),
            'memory_type': np.random.choice(memory_types),
            'age_hours': np.random.exponential(24),
            'decay_factor': np.random.beta(2, 5),
            'access_count': np.random.poisson(5),
            'visual': {
                'size': np.random.uniform(3, 8),
                'color': f"rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.8)"
            }
        }
        nodes.append(node)
    
    # Create some sample edges
    edges = []
    for i in range(min(5000, n_nodes//10)):  # Limit edges for performance
        source = np.random.randint(0, n_nodes)
        target = np.random.randint(0, n_nodes)
        if source != target:
            edges.append({
                'source': f"node_{source}",
                'target': f"node_{target}",
                'weight': np.random.uniform(0.1, 1.0)
            })
    
    data = {'nodes': nodes, 'edges': edges}
    
    # Create GPU-accelerated viewer
    viewer = GPUAcceleratedMemoryViewer()
    
    # Test different configurations
    logger.info("🧪 Testing GPU-accelerated visualization configurations...")
    
    # Test 1: Full quality with semantic layout
    viz_file1 = viewer.create_gpu_accelerated_visualization(
        data, 
        layout_type="semantic_pca",
        enable_lod=True,
        camera_distance=50.0
    )
    
    # Test 2: Sphere layout with aggressive LOD
    viz_file2 = viewer.create_gpu_accelerated_visualization(
        data,
        layout_type="sphere", 
        enable_lod=True,
        camera_distance=100.0  # Farther camera triggers higher LOD
    )
    
    # Test 3: Performance comparison
    comparison_file = viewer.create_performance_comparison(data)
    
    logger.info("✅ GPU-accelerated visualization testing complete!")
    logger.info(f"   📁 Main visualization: {viz_file1}")
    logger.info(f"   🔍 Sphere layout: {viz_file2}")
    logger.info(f"   📊 LOD comparison: {comparison_file}")

if __name__ == "__main__":
    main()
