#!/usr/bin/env python3
"""
merX Memory 3D Visualization - Streamlit Interactive Dashboard

A beautiful, interactive web interface for exploring merX memory structures
with real-time 3D visualizations, memory decay analysis, and exploration tools.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="merX Memory 3D Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for beautiful styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .memory-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .fractal-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .tree-header {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .network-header {
        background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from enhanced_3d_viewer import MemoryVisualization3D
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False
    st.error("Could not import enhanced_3d_viewer. Please check your setup.")

# Initialize session state
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "All"
if 'decay_threshold' not in st.session_state:
    st.session_state.decay_threshold = 0.0


def load_or_generate_data():
    """Load or generate visualization data."""
    if st.session_state.viz_data is None and VIEWER_AVAILABLE:
        with st.spinner("🧠 Loading memory data..."):
            viz = MemoryVisualization3D()
            
            # Try to setup real data, fallback to sample
            if viz.setup_memory_engine():
                st.success("✅ Connected to real merX memory data")
            else:
                st.info("🌱 Using sample data for demonstration")
            
            st.session_state.viz_data = viz.load_memory_data()
            st.session_state.viz_obj = viz
    
    return st.session_state.viz_data


def filter_data(data, domain_filter, decay_threshold):
    """Filter data based on user selections."""
    if not data:
        return data
    
    filtered_nodes = []
    for node in data["nodes"]:
        # Domain filter
        if domain_filter != "All" and node["domain"] != domain_filter:
            continue
        
        # Decay threshold filter
        if node["decay_factor"] < decay_threshold:
            continue
        
        filtered_nodes.append(node)
    
    # Filter edges to only include nodes that passed the filter
    node_ids = {node["id"] for node in filtered_nodes}
    filtered_edges = [
        edge for edge in data["edges"]
        if edge["source"] in node_ids and edge["target"] in node_ids
    ]
    
    return {
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "stats": data["stats"]  # Keep original stats for reference
    }


def create_dashboard_header():
    """Create the main dashboard header."""
    st.markdown("""
    <div class="memory-card">
        <h1 style="margin: 0; font-size: 2.5rem;">🧠 merX Memory 3D Visualizer</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Explore your memory structures in beautiful 3D space with decay visualization
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar_controls(data):
    """Create sidebar controls for filtering and options."""
    st.sidebar.markdown("## 🎛️ Visualization Controls")
    
    # Domain filter
    if data and data["nodes"]:
        domains = ["All"] + list(set(node["domain"] for node in data["nodes"]))
        st.session_state.selected_domain = st.sidebar.selectbox(
            "🏷️ Filter by Domain",
            domains,
            index=domains.index(st.session_state.selected_domain) if st.session_state.selected_domain in domains else 0
        )
    
    # Decay threshold slider
    st.session_state.decay_threshold = st.sidebar.slider(
        "💫 Minimum Memory Freshness",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.decay_threshold,
        step=0.1,
        help="Filter out memories below this decay threshold (0 = show all, 1 = only fresh)"
    )
    
    st.sidebar.markdown("---")
    
    # Visualization options
    st.sidebar.markdown("## 🎨 Display Options")
    
    show_connections = st.sidebar.checkbox("🔗 Show Connections", value=True)
    show_labels = st.sidebar.checkbox("🏷️ Show Labels", value=False)
    animation_speed = st.sidebar.slider("⚡ Animation Speed", 0.5, 3.0, 1.0, 0.5)
    
    st.sidebar.markdown("---")
    
    # Memory insights
    if data:
        st.sidebar.markdown("## 📊 Memory Insights")
        
        total_memories = len(data["nodes"])
        fresh_memories = len([n for n in data["nodes"] if n["decay_factor"] > 0.8])
        fading_memories = len([n for n in data["nodes"] if n["decay_factor"] < 0.3])
        
        st.sidebar.metric("Total Memories", f"{total_memories:,}")
        st.sidebar.metric("Fresh Memories", f"{fresh_memories:,}", f"{fresh_memories/total_memories*100:.1f}%")
        st.sidebar.metric("Fading Memories", f"{fading_memories:,}", f"-{fading_memories/total_memories*100:.1f}%")
        
        # Domain distribution
        if data["stats"]["domains"]:
            st.sidebar.markdown("### 🏷️ Domain Distribution")
            domain_df = pd.DataFrame([
                {"Domain": k, "Count": v} 
                for k, v in data["stats"]["domains"].items()
            ])
            st.sidebar.bar_chart(domain_df.set_index("Domain"))
    
    return show_connections, show_labels, animation_speed


def create_3d_network_viz(filtered_data, show_connections, show_labels):
    """Create the 3D network visualization."""
    if not filtered_data or not filtered_data["nodes"]:
        st.warning("No data available for visualization. Try adjusting your filters.")
        return
    
    st.markdown("""
    <div class="network-header">
        <h2 style="margin: 0;">🌌 3D Memory Network</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Explore connections between memories in 3D space
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 3D visualization using the enhanced viewer
    if VIEWER_AVAILABLE:
        viz_file = st.session_state.viz_obj.create_3d_memory_network(filtered_data)
        
        # Read the HTML file and display it
        if os.path.exists(viz_file):
            with open(viz_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Display in iframe or as HTML component
            st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.error("Failed to create 3D visualization")
    else:
        st.error("Enhanced 3D viewer not available")


def create_tree_forest_viz(filtered_data):
    """Create the tree-of-trees visualization."""
    if not filtered_data or not filtered_data["nodes"]:
        return
    
    st.markdown("""
    <div class="tree-header">
        <h2 style="margin: 0;">🌲 Memory Forest - Tree of Trees</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Domain-based hierarchical memory structures
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if VIEWER_AVAILABLE:
        viz_file = st.session_state.viz_obj.create_tree_of_trees_3d(filtered_data)
        
        if os.path.exists(viz_file):
            with open(viz_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.error("Failed to create tree visualization")


def create_fractal_viz(filtered_data):
    """Create the fractal hierarchy visualization."""
    if not filtered_data or not filtered_data["nodes"]:
        return
    
    st.markdown("""
    <div class="fractal-header">
        <h2 style="margin: 0;">🌀 Fractal Memory Hierarchy</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Interactive clickable exploration of memory patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if VIEWER_AVAILABLE:
        viz_file = st.session_state.viz_obj.create_fractal_hierarchy_graph(filtered_data)
        
        if os.path.exists(viz_file):
            with open(viz_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.error("Failed to create fractal visualization")


def create_analytics_dashboard(data):
    """Create analytics dashboard with charts and metrics."""
    if not data or not data["nodes"]:
        return
    
    st.markdown("## 📊 Memory Analytics Dashboard")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_nodes = len(data["nodes"])
    total_edges = len(data["edges"])
    avg_decay = np.mean([node["decay_factor"] for node in data["nodes"]])
    domains_count = len(set(node["domain"] for node in data["nodes"]))
    
    with col1:
        st.metric("Total Memories", f"{total_nodes:,}")
    
    with col2:
        st.metric("Connections", f"{total_edges:,}")
    
    with col3:
        st.metric("Average Freshness", f"{avg_decay:.3f}")
    
    with col4:
        st.metric("Domains", domains_count)
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Decay distribution histogram
        decay_values = [node["decay_factor"] for node in data["nodes"]]
        fig_decay = px.histogram(
            x=decay_values,
            nbins=20,
            title="Memory Decay Distribution",
            labels={"x": "Decay Factor", "y": "Count"},
            color_discrete_sequence=["#3742fa"]
        )
        fig_decay.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_decay, use_container_width=True)
    
    with col2:
        # Memory type distribution
        if data["stats"]["memory_types"]:
            memory_types_df = pd.DataFrame([
                {"Type": k, "Count": v} 
                for k, v in data["stats"]["memory_types"].items()
            ])
            
            fig_types = px.pie(
                memory_types_df,
                values="Count",
                names="Type",
                title="Memory Types Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_types.update_layout(height=400)
            st.plotly_chart(fig_types, use_container_width=True)
    
    # Age vs Decay scatter plot
    if data["nodes"]:
        scatter_data = []
        for node in data["nodes"]:
            scatter_data.append({
                "Age (hours)": node["age_hours"],
                "Decay Factor": node["decay_factor"],
                "Domain": node["domain"],
                "Memory Type": node["memory_type"],
                "Content": node["content"][:50] + "..."
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig_scatter = px.scatter(
            scatter_df,
            x="Age (hours)",
            y="Decay Factor",
            color="Domain",
            size="Decay Factor",
            hover_data=["Memory Type", "Content"],
            title="Memory Age vs Decay Analysis",
            log_x=True
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)


def create_memory_explorer(data):
    """Create interactive memory explorer."""
    if not data or not data["nodes"]:
        return
    
    st.markdown("## 🔍 Memory Explorer")
    
    # Search functionality
    search_query = st.text_input("🔎 Search Memories", placeholder="Enter keywords to search...")
    
    # Filter nodes based on search
    if search_query:
        matching_nodes = [
            node for node in data["nodes"]
            if search_query.lower() in node["content"].lower()
            or search_query.lower() in node["domain"].lower()
            or any(search_query.lower() in tag.lower() for tag in node.get("tags", []))
        ]
    else:
        matching_nodes = data["nodes"][:20]  # Show first 20 by default
    
    st.write(f"Found {len(matching_nodes)} matching memories")
    
    # Display matching memories
    for i, node in enumerate(matching_nodes[:10]):  # Limit to 10 for display
        with st.expander(f"Memory {i+1}: {node['content'][:60]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Domain:** {node['domain']}")
                st.write(f"**Type:** {node['memory_type']}")
                st.write(f"**Age:** {node['age_hours']:.1f} hours")
                
            with col2:
                st.write(f"**Decay Factor:** {node['decay_factor']:.3f}")
                st.write(f"**Access Count:** {node['access_count']}")
                st.write(f"**Tags:** {', '.join(node.get('tags', []))}")
            
            st.write(f"**Content:** {node['content']}")
            
            # Visual decay indicator
            decay_color = "green" if node['decay_factor'] > 0.7 else "orange" if node['decay_factor'] > 0.4 else "red"
            st.markdown(f"**Freshness:** <span style='color: {decay_color}'>{'🟢' * int(node['decay_factor'] * 5)}{'⚪' * (5 - int(node['decay_factor'] * 5))}</span>", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    # Create header
    create_dashboard_header()
    
    # Load data
    data = load_or_generate_data()
    
    if not data:
        st.error("Could not load memory data. Please check your setup.")
        return
    
    # Create sidebar controls
    show_connections, show_labels, animation_speed = create_sidebar_controls(data)
    
    # Filter data based on user selections
    filtered_data = filter_data(data, st.session_state.selected_domain, st.session_state.decay_threshold)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌌 3D Network", 
        "🌲 Tree Forest", 
        "🌀 Fractal", 
        "📊 Analytics",
        "🔍 Explorer"
    ])
    
    with tab1:
        create_3d_network_viz(filtered_data, show_connections, show_labels)
    
    with tab2:
        create_tree_forest_viz(filtered_data)
    
    with tab3:
        create_fractal_viz(filtered_data)
    
    with tab4:
        create_analytics_dashboard(filtered_data)
    
    with tab5:
        create_memory_explorer(filtered_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 2rem;">
        🧠 merX Memory 3D Visualizer | Built with Streamlit & Plotly<br>
        🌿 Explore your memories • 💫 Understand decay • 🔗 Discover connections
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
