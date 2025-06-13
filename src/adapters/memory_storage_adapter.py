"""
Memory Storage Adapter for merX - Provides compatibility between different storage implementations.

This adapter allows the system to use enhanced storage components like MemoryIOOrchestrator
while maintaining compatibility with code that expects the original IMemoryStorage interface.

Features:
- Compatible with the original IMemoryStorage interface
- Optimized performance through direct RAMX access
- Support for distributed storage with large datasets
- Compression utilities for efficient storage
"""

import logging
import time
from typing import Dict, List, Optional, Any, BinaryIO, Union, Set
from uuid import UUID
import threading

from src.interfaces import MemoryNode, IMemoryStorage, MemoryLink
from src.core.memory_io_orchestrator import MemoryIOOrchestrator
from src.core.ramx import RAMXNode

logger = logging.getLogger(__name__)


class MemoryStorageAdapter(IMemoryStorage):
    """
    Adapter that makes MemoryIOOrchestrator compatible with the IMemoryStorage interface.
    
    This allows existing code to use the enhanced components with minimal changes.
    """
    
    def __init__(self, orchestrator: MemoryIOOrchestrator):
        """
        Initialize the adapter with a MemoryIOOrchestrator.
        
        Args:
            orchestrator: The MemoryIOOrchestrator to adapt
        """
        self.orchestrator = orchestrator
        logger.info("Initialized Memory Storage Adapter")
    
    def append_node(self, node: MemoryNode) -> int:
        """
        Append a node to storage. Returns byte offset.
        
        This is an adapter method that maps to MemoryIOOrchestrator's insert_node.
        The original interface returns an int offset, but since RAMX architecture 
        doesn't use offsets in the same way, we return 0 as a placeholder.
        """
        self.orchestrator.insert_node(node)
        logger.debug(f"Adapted append_node call to insert_node for {node.id}")
        return 0  # Placeholder as RAMX doesn't use offsets like the disk-based storage
    
    def read_node_by_id(self, node_id: UUID) -> Optional[MemoryNode]:
        """Read a node by its UUID."""
        return self.orchestrator.get_node(node_id)
    
    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        """Get a node by its UUID (alias for read_node_by_id)."""
        return self.read_node_by_id(node_id)
    
    def read_node(self, offset: int) -> Optional[MemoryNode]:
        """
        Read a node at the given byte offset.
        
        Note: Since RAMX doesn't use byte offsets, this method is not fully
        functional in this adapter. It will return None.
        """
        logger.warning("read_node by offset not supported in RAMX architecture")
        return None
    
    def update_node(self, node: MemoryNode) -> bool:
        """Update an existing node."""
        return self.orchestrator.update_node(node)
    
    def get_all_nodes(self) -> List[MemoryNode]:
        """Get all nodes in the storage."""
        # Get from RAMX directly for efficiency
        ramx_nodes = self.orchestrator.ramx.get_all_nodes()
        return [node.to_memory_node() for node in ramx_nodes]
    
    def get_nodes_by_type(self, node_type: str, limit: int = 100) -> List[MemoryNode]:
        """Get nodes by type."""
        ramx_nodes = self.orchestrator.ramx.recall_by_type(node_type, limit)
        return [node.to_memory_node() for node in ramx_nodes]
    
    def apply_global_decay(self) -> int:
        """Apply decay to all nodes in storage."""
        return self.orchestrator.ramx.apply_global_decay()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage system."""
        return self.orchestrator.get_stats()
    def serialize_node(self, node: MemoryNode) -> bytes:
        """Serialize a node to binary format."""
        return self.orchestrator.serialize_to_bytes(node)
    
    def append_node_2(self, node: MemoryNode) -> Optional[UUID]:
        """Append a memory node to storage."""
        return self.orchestrator.insert_node(node)
    
    def _2(self, node_id: UUID) -> Optional[MemoryNode]:
        """Read a node by its ID."""
        return self.orchestrator.get_node(node_id)
    
    def update_node_activation(self, node_id: UUID, new_activation: float) -> bool:
        """Update a node's activation value."""
        node = self.orchestrator.get_node(node_id)
        if node:
            node.activation = new_activation
            return self.orchestrator.update_node(node)
        return False
    
    def get_related_nodes(self, node_id: UUID, max_depth: int = 1) -> List[MemoryNode]:
        """Get nodes related to a specific node."""
        # Use RAMX for optimized graph traversal if available
        if hasattr(self.orchestrator, 'ramx'):
            ramx_nodes = self.orchestrator.ramx.get_related_nodes(node_id, max_depth)
            return [node.to_memory_node() for node in ramx_nodes]
            
        # Fallback implementation
        related_ids = set()
        to_process = [(node_id, 0)]  # (id, depth)
        processed = set()
        
        while to_process:
            current_id, depth = to_process.pop(0)
            
            if current_id in processed:
                continue
                
            processed.add(current_id)
            
            node = self.orchestrator.get_node(current_id)
            if not node:
                continue
                
            if current_id != node_id:  # Don't add the starting node
                related_ids.add(current_id)
                
            # Stop at max depth
            if depth >= max_depth:
                continue
                
            # Add linked nodes to processing queue
            for link in node.links:
                if link.to_id not in processed:
                    to_process.append((link.to_id, depth + 1))
        
        # Get actual nodes
        related_nodes = []
        for related_id in related_ids:
            node = self.orchestrator.get_node(related_id)
            if node:
                related_nodes.append(node)
                
        return related_nodes
    
    def get_nodes_by_activation(self, min_activation: float = 0.0, limit: int = 100) -> List[MemoryNode]:
        """Get nodes with activation above the specified threshold."""
        if hasattr(self.orchestrator, 'ramx'):
            ramx_nodes = self.orchestrator.ramx.get_nodes_by_activation(min_activation, limit)
            return [node.to_memory_node() for node in ramx_nodes]
            
        # Fallback implementation
        all_nodes = self.get_all_nodes()
        activated_nodes = [node for node in all_nodes if node.activation >= min_activation]
        activated_nodes.sort(key=lambda x: x.activation, reverse=True)
        return activated_nodes[:limit]
    
    def get_nodes_by_tags(self, tags: List[str], match_all: bool = False, limit: int = 100) -> List[MemoryNode]:
        """Get nodes that have the specified tags."""
        if hasattr(self.orchestrator, 'ramx'):
            ramx_nodes = self.orchestrator.ramx.get_nodes_by_tags(tags, match_all, limit)
            return [node.to_memory_node() for node in ramx_nodes]
            
        # Fallback implementation
        all_nodes = self.get_all_nodes()
        
        if match_all:
            # Node must have all tags
            matching_nodes = [
                node for node in all_nodes
                if all(tag.lower() in [t.lower() for t in node.tags] for tag in tags)
            ]
        else:
            # Node must have at least one tag
            matching_nodes = [
                node for node in all_nodes
                if any(tag.lower() in [t.lower() for t in node.tags] for tag in tags)
            ]
            
        # Sort by activation
        matching_nodes.sort(key=lambda x: x.activation, reverse=True)
        return matching_nodes[:limit]
        
    def bulk_insert(self, nodes: List[MemoryNode]) -> List[UUID]:
        """Insert multiple nodes at once for improved performance."""
        return [self.orchestrator.insert_node(node) for node in nodes]
    
    def flush(self) -> None:
        """Flush any in-memory changes to disk."""
        if hasattr(self.orchestrator, 'flush'):
            self.orchestrator.flush()
            
    def optimize(self) -> Dict[str, Any]:
        """Optimize storage for improved performance."""
        if hasattr(self.orchestrator, 'optimize'):
            return self.orchestrator.optimize()
        return {"optimized": False, "reason": "Operation not supported"}
        
    @property
    def ramx(self):
        """Direct access to RAMX for optimized operations."""
        if hasattr(self.orchestrator, 'ramx'):
            return self.orchestrator.ramx
        return None
