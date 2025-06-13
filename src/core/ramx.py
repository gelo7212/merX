"""
RAMX - High-performance RAM-based memory store for merX.
"""
import time
import logging
import threading
import re
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from src.interfaces import MemoryNode, MemoryLink

logger = logging.getLogger(__name__)

@dataclass
class RAMXNode:
    """RAM-optimized memory node for high-performance operations."""
    id: UUID
    content: str
    node_type: str
    timestamp: float  # Unix timestamp for faster calculations
    activation: float
    decay_rate: float
    version: int = 1
    version_of: Optional[UUID] = None
    links: Dict[UUID, Tuple[float, str]] = field(default_factory=dict)  # node_id -> (weight, link_type)
    tags: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # For vector similarity search
    
    def add_link(self, to_id: UUID, weight: float = 0.5, link_type: str = "default") -> None:
        """Add or update a link to another node."""
        self.links[to_id] = (weight, link_type)
    
    def get_links(self) -> List[Tuple[UUID, float, str]]:
        """Get all links as a list of (id, weight, type) tuples."""
        return [(node_id, weight, link_type) for node_id, (weight, link_type) in self.links.items()]
    
    def get_link_weight(self, node_id: UUID) -> float:
        """Get the weight of a link to a specific node."""
        if node_id in self.links:
            return self.links[node_id][0]
        return 0.0
    
    def decay(self, current_time: float) -> float:
        """Apply decay based on time elapsed since creation."""
        if self.activation <= 0:
            return 0.0
        
        age = current_time - self.timestamp
        # Exponential decay formula: A = A0 * e^(-decay_rate * time)
        self.activation *= math.exp(-self.decay_rate * age)
        self.activation = max(0.0, min(1.0, self.activation))  # Clamp between 0 and 1
        return self.activation
    
    def boost(self, amount: float = 0.1) -> float:
        """Boost the activation of this node."""
        self.activation = min(1.0, self.activation + amount * (1.0 - self.activation))
        return self.activation
    
    def to_memory_node(self) -> MemoryNode:
        """Convert to standard MemoryNode format."""
        links = [
            MemoryLink(to_id=node_id, weight=weight, link_type=link_type)
            for node_id, (weight, link_type) in self.links.items()
        ]
        
        return MemoryNode(
            id=self.id,
            content=self.content,
            node_type=self.node_type,
            version=self.version,
            timestamp=datetime.fromtimestamp(self.timestamp),
            activation=self.activation,
            decay_rate=self.decay_rate,
            version_of=self.version_of,
            links=links,
            tags=self.tags.copy()
    
        )
    @staticmethod
    def from_memory_node(node: MemoryNode) -> 'RAMXNode':
        """Create a RAMXNode from a standard MemoryNode."""
        links = {}
        if node.links:
            links = {
                link.to_id: (link.weight, link.link_type)
                for link in node.links
            }
        
        tags = node.tags[:] if node.tags else []
        
        return RAMXNode(
            id=node.id,
            content=node.content,
            node_type=node.node_type,
            timestamp=node.timestamp.timestamp(),
            activation=node.activation,
            decay_rate=node.decay_rate,
            version=node.version,
            version_of=node.version_of,
            links=links,
            tags=tags
        )


class RAMX:
    """
    High-performance RAM-based memory store with neural-like activation and spreading.
    
    Features:
    - In-memory storage for fast access
    - Neural-like spreading activation for querying
    - Time-based decay
    - Automated node eviction when memory pressure is high
    - Thread-safe operations
    """
    
    def __init__(self, 
                capacity: int = 100000,  # Default max capacity: 100K nodes
                activation_threshold: float = 0.01,
                spreading_decay: float = 0.7,
                max_hops: int = 3):
        """
        Initialize the RAMX memory system.
        
        Args:
            capacity: Maximum number of nodes to keep in RAM
            activation_threshold: Minimum activation level to keep
            spreading_decay: Factor for decaying activation during spreading
            max_hops: Maximum number of hops for spreading activation
        """
        self._nodes: Dict[UUID, RAMXNode] = {}
        self._node_lock = threading.RLock()
        self._capacity = capacity
        self._activation_threshold = activation_threshold
        self._spreading_decay = spreading_decay
        self._max_hops = max_hops
        self._word_index: Dict[str, Set[UUID]] = {}  # Word -> set of node IDs containing it
        self._tag_index: Dict[str, Set[UUID]] = {}   # Tag -> set of node IDs with this tag
        self._type_index: Dict[str, Set[UUID]] = {}  # Node type -> set of node IDs of this type
        
        # Background decay thread
        self._decay_stop = threading.Event()
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()
        
        logger.info(f"Initialized RAMX with capacity {capacity}")
    
    def add_node(self, node: RAMXNode) -> None:
        """Add a new node to the RAM store."""
        with self._node_lock:
            # Check capacity before adding
            if len(self._nodes) >= self._capacity:
                self._evict_nodes()
            
            # Add to main node store
            self._nodes[node.id] = node
            
            # Index content words
            self._index_content(node)
            
            # Index tags
            self._index_tags(node)
            
            # Index node type
            self._index_type(node)
    
    def get_node(self, node_id: UUID) -> Optional[RAMXNode]:
        """Get a node by its ID."""
        with self._node_lock:
            node = self._nodes.get(node_id)
            if node:
                # Boost activation when accessed
                node.boost(0.05)
            return node
    
    def add_or_update_node(self, node: RAMXNode) -> None:
        """Add a new node or update existing one."""
        with self._node_lock:
            existing = self._nodes.get(node.id)
            if existing:
                # Preserve higher activation value if present
                node.activation = max(node.activation, existing.activation)
                
                # Remove old indexes
                self._remove_from_indexes(existing)
            
            # Add the node
            self.add_node(node)
    
    def add_memory_node(self, node: MemoryNode) -> UUID:
        """Convert and add a standard MemoryNode."""
        ram_node = RAMXNode.from_memory_node(node)
        self.add_node(ram_node)
        return ram_node.id
    
    def _index_content(self, node: RAMXNode) -> None:
        """Index the node's content by words."""
        # Simple word extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', node.content.lower())
        for word in words:
            if word not in self._word_index:
                self._word_index[word] = set()
            self._word_index[word].add(node.id)
    
    def _index_tags(self, node: RAMXNode) -> None:
        """Index the node's tags."""
        for tag in node.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(node.id)
    
    def _index_type(self, node: RAMXNode) -> None:
        """Index the node by type."""
        if node.node_type not in self._type_index:
            self._type_index[node.node_type] = set()
        self._type_index[node.node_type].add(node.id)
    
    def _remove_from_indexes(self, node: RAMXNode) -> None:
        """Remove a node from all indexes."""
        # Remove from word index
        words = re.findall(r'\b\w+\b', node.content.lower())
        for word in words:
            if word in self._word_index and node.id in self._word_index[word]:
                self._word_index[word].remove(node.id)
        
        # Remove from tag index
        for tag in node.tags:
            tag_lower = tag.lower()
            if tag_lower in self._tag_index and node.id in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].remove(node.id)
        
        # Remove from type index
        if node.node_type in self._type_index and node.id in self._type_index[node.node_type]:
            self._type_index[node.node_type].remove(node.id)
    
    def _evict_nodes(self) -> None:
        """Evict least activated nodes when capacity is reached."""
        # Sort nodes by activation (ascending)
        sorted_nodes = sorted(self._nodes.items(), key=lambda x: x[1].activation)
        
        # Evict bottom 10% of nodes
        evict_count = max(1, len(self._nodes) // 10)
        for i in range(evict_count):
            if i < len(sorted_nodes):
                node_id, node = sorted_nodes[i]
                if node.activation < self._activation_threshold:
                    self._remove_from_indexes(node)
                    del self._nodes[node_id]
                    logger.debug(f"Evicted node {node_id} with activation {node.activation:.3f}")
    
    def _decay_loop(self) -> None:
        """Background thread for applying decay to nodes."""
        while not self._decay_stop.is_set():
            try:
                # Apply decay every 60 seconds
                time.sleep(60)
                self.apply_global_decay()
            except Exception as e:
                logger.error(f"Error in decay loop: {e}")
    
    def apply_global_decay(self) -> int:
        """Apply decay to all nodes in memory. Returns number processed."""
        current_time = time.time()
        processed = 0
        
        with self._node_lock:
            for node in self._nodes.values():
                node.decay(current_time)
                processed += 1
        
        logger.debug(f"Applied decay to {processed} nodes")
        return processed
    
    def shutdown(self) -> None:
        """Shutdown the RAMX system cleanly."""
        # Stop background decay thread
        self._decay_stop.set()
        self._decay_thread.join(timeout=2.0)
        logger.info("RAMX shutdown complete")
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in RAM storage."""
        with self._node_lock:
            return len(self._nodes)
    
    def get_all_nodes(self) -> List[RAMXNode]:
        """Get all nodes in memory."""
        with self._node_lock:
            return list(self._nodes.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        with self._node_lock:
            total_nodes = len(self._nodes)
            active_nodes = sum(1 for node in self._nodes.values() if node.activation > self._activation_threshold)
            avg_activation = sum(node.activation for node in self._nodes.values()) / max(1, total_nodes)
            
            type_counts = {}
            for node_type, nodes in self._type_index.items():
                type_counts[node_type] = len(nodes)
            
            return {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "average_activation": avg_activation,
                "node_types": type_counts,
                "word_index_size": len(self._word_index),
                "tag_index_size": len(self._tag_index),
                "memory_usage_percentage": (total_nodes / self._capacity) * 100
            }
    
    # --- ADVANCED RETRIEVAL WITH TRIGGER LOGIC ---
    
    def trigger_based_recall(self, query: str, limit: int = 10, hop_boost_factor: float = 0.5) -> List[Tuple[RAMXNode, float]]:
        """
        Neural-like triggering recall based on a text query.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            hop_boost_factor: How much to boost activation per hop
            
        Returns:
            List of (node, score) tuples, sorted by descending score
        """
        with self._node_lock:
            # Find seed nodes that match the query directly
            seed_nodes = self._find_seed_nodes(query)
            if not seed_nodes:
                return []
            
            # Track activation levels during spreading
            activation_map: Dict[UUID, float] = {}
            
            # Initialize with seed nodes
            for node_id, initial_score in seed_nodes:
                activation_map[node_id] = initial_score
            
            # Spreading activation through the network
            self._spread_activation(activation_map, hop_boost_factor)
            
            # Retrieve the actual nodes and sort by activation
            results = []
            for node_id, activation in sorted(activation_map.items(), key=lambda x: x[1], reverse=True):
                if node_id in self._nodes:
                    results.append((self._nodes[node_id], activation))
                    if len(results) >= limit:
                        break
            
            # Boost the nodes that were retrieved
            for node, _ in results:
                node.boost(0.05)
            
            return results
    
    def _find_seed_nodes(self, query: str) -> List[Tuple[UUID, float]]:
        """Find initial seed nodes matching the query."""
        query_words = re.findall(r'\b\w+\b', query.lower())
        
        # Calculate scores for each node based on word matches
        node_scores: Dict[UUID, float] = {}
        
        for word in query_words:
            if word in self._word_index:
                for node_id in self._word_index[word]:
                    # Increase score for each matching word
                    if node_id not in node_scores:
                        node_scores[node_id] = 0.0
                    node_scores[node_id] += 1.0 / max(1, len(query_words))
        
        # Normalize scores to 0-1 range
        if node_scores:
            max_score = max(node_scores.values())
            if max_score > 0:
                node_scores = {k: v / max_score for k, v in node_scores.items()}
        
        # Return sorted by score
        return sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    
    def spreading_activation(self, start_nodes: List[UUID], max_depth: int = 3) -> Dict[UUID, float]:
        """
        Perform spreading activation from start nodes.
        
        Args:
            start_nodes: List of node IDs to start activation from
            max_depth: Maximum number of hops to spread activation
            
        Returns:
            Dictionary mapping node IDs to activation values
        """
        with self._node_lock:
            # Initialize activation map with start nodes
            activation_map: Dict[UUID, float] = {node_id: 1.0 for node_id in start_nodes}
            
            # Track visited nodes to avoid cycles
            visited: Set[UUID] = set(start_nodes)
            
            # Spread for specified number of hops
            for hop in range(max_depth):
                next_nodes = []
                hop_decay = self._spreading_decay ** (hop + 1)
                
                # For each currently active node
                for node_id in list(visited):
                    if node_id not in self._nodes:
                        continue
                    
                    # Get all connected nodes
                    node = self._nodes[node_id]
                    for linked_id, (weight, _) in node.links.items():
                        if linked_id not in self._nodes:
                            continue
                        
                        # Calculate spread activation
                        spread_value = activation_map.get(node_id, 0) * weight * hop_decay
                        
                        # Only spread if significant
                        if spread_value < self._activation_threshold:
                            continue
                        
                        # Add or update activation
                        if linked_id in activation_map:
                            activation_map[linked_id] = max(activation_map[linked_id], spread_value)
                        else:
                            activation_map[linked_id] = spread_value
                            next_nodes.append(linked_id)
                
                # Add newly activated nodes to visited set
                visited.update(next_nodes)
                
                # Stop if no more spreading
                if not next_nodes:
                    break
            
            # Filter out nodes below threshold
            return {k: v for k, v in activation_map.items() if v >= self._activation_threshold}
    
    def spreading_activation2(self, start_nodes: List[UUID], max_depth: int = 2) -> Dict[UUID, float]:
        """
        Perform spreading activation from start nodes.
        
        Args:
            start_nodes: List of node IDs to start activation from
            max_depth: Maximum number of hops to spread
            
        Returns:
            Dictionary mapping node IDs to activation values
        """
        # Initialize activation map with start nodes at full activation
        activation_map = {node_id: 1.0 for node_id in start_nodes}
        
        # Set max hops based on provided depth
        old_max_hops = self._max_hops
        self._max_hops = max_depth
        
        try:
            # Use the existing _spread_activation implementation
            self._spread_activation(activation_map, hop_boost_factor=1.0)
            return activation_map
        finally:
            # Restore original max_hops
            self._max_hops = old_max_hops
            
    def _spread_activation(self, activation_map: Dict[UUID, float], hop_boost_factor: float) -> None:
        """Spread activation through the network."""
        # Start with the initially activated nodes
        current_nodes = list(activation_map.keys())
        
        # Track visited nodes to avoid cycles
        visited: Set[UUID] = set(current_nodes)
        
        # Spread for multiple hops
        for hop in range(self._max_hops):
            next_nodes = []
            hop_decay = self._spreading_decay ** (hop + 1)
            
            for node_id in current_nodes:
                if node_id not in self._nodes:
                    continue
                
                node = self._nodes[node_id]
                parent_activation = activation_map.get(node_id, 0.0)
                
                # Spread to linked nodes
                for linked_id, (weight, _) in node.links.items():
                    if linked_id in visited:
                        continue
                    
                    # Calculate activation to pass along this connection
                    passed_activation = parent_activation * weight * hop_decay
                    
                    # Only spread if meaningful activation
                    if passed_activation > self._activation_threshold:
                        if linked_id not in activation_map:
                            activation_map[linked_id] = 0.0
                        
                        # Add activation with boost factor
                        activation_map[linked_id] += passed_activation * hop_boost_factor
                        next_nodes.append(linked_id)
                        visited.add(linked_id)
            
            # If no more spreading, stop
            if not next_nodes:
                break
            
            current_nodes = next_nodes
    
    def recall_by_tags(self, tags: List[str], limit: int = 10) -> List[RAMXNode]:
        """Recall memories by matching tags."""
        if not tags:
            return []
        
        with self._node_lock:
            # Find nodes with matching tags
            matching_node_sets = []
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower in self._tag_index:
                    matching_node_sets.append(self._tag_index[tag_lower])
            
            # No matches found
            if not matching_node_sets:
                return []
            
            # Start with nodes matching the first tag
            if matching_node_sets:
                result_set = matching_node_sets[0]
                
                # Combine with other tags (intersection for AND logic)
                for node_set in matching_node_sets[1:]:
                    result_set = result_set.intersection(node_set)
            else:
                result_set = set()
            
            # Convert to nodes
            results = []
            for node_id in result_set:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node.boost(0.05)  # Boost activation
                    results.append(node)
                    if len(results) >= limit:
                        break
            
            # Sort by activation
            results.sort(key=lambda x: x.activation, reverse=True)
            return results
    
    def recall_by_type(self, node_type: str, limit: int = 10) -> List[RAMXNode]:
        """Recall memories by node type."""
        with self._node_lock:
            if node_type not in self._type_index:
                return []
            
            results = []
            for node_id in self._type_index[node_type]:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node.boost(0.02)  # Small boost
                    results.append(node)
                    if len(results) >= limit:
                        break
            
            # Sort by activation
            results.sort(key=lambda x: x.activation, reverse=True)
            return results
    
    def get_related_nodes(self, node_id: UUID, min_weight: float = 0.0) -> List[Tuple[RAMXNode, float]]:
        """Get nodes directly related to the given node."""
        with self._node_lock:
            if node_id not in self._nodes:
                return []
            
            node = self._nodes[node_id]
            related = []
            
            for related_id, (weight, _) in node.links.items():
                if weight >= min_weight and related_id in self._nodes:
                    related.append((self._nodes[related_id], weight))
            
            # Sort by weight
            related.sort(key=lambda x: x[1], reverse=True)
            return related
