"""
Distributed Storage Module for merX.

Enables distributed memory storage across multiple files or machines
to support extremely large datasets (100K+ records).
"""

import os
import logging
import threading
import time
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from datetime import datetime
from uuid import UUID
import hashlib
import json
from dataclasses import asdict, is_dataclass

from src.interfaces import MemoryNode
from src.core.memory_io_orchestrator import MemoryIOOrchestrator
from src.core.ramx import RAMX, RAMXNode
from src.core.enhanced_index_manager import EnhancedIndexManager
from src.utils.compression import MemoryCompression

logger = logging.getLogger(__name__)


class ShardManager:
    """
    Manages sharding of memory nodes across multiple files.
    
    Uses consistent hashing to distribute nodes evenly and minimize
    resharding when adding new storage locations.
    """
    
    def __init__(self, shard_count: int = 4, replicas: int = 1):
        """
        Initialize the shard manager.
        
        Args:
            shard_count: Number of shards to distribute data across
            replicas: Number of replicas per node for redundancy
        """
        self.shard_count = shard_count
        self.replicas = replicas
        self.hash_ring: Dict[int, str] = {}  # Hash position -> shard ID
        self.shards: Dict[str, str] = {}     # Shard ID -> shard path
        
        # Lock for concurrent access
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ShardManager with {shard_count} shards and {replicas} replicas")
    
    def add_shard(self, shard_id: str, shard_path: str) -> None:
        """
        Add a shard to the hash ring.
        
        Args:
            shard_id: Unique ID for the shard
            shard_path: Path to the shard data file
        """
        with self._lock:
            # Store shard path
            self.shards[shard_id] = shard_path
            
            # Add shard to ring with multiple virtual points for better distribution
            for i in range(self.replicas * 100):  # Virtual points
                key = f"{shard_id}:{i}"
                hash_val = self._hash_key(key)
                self.hash_ring[hash_val] = shard_id
            
            logger.info(f"Added shard {shard_id} at {shard_path}")
    
    def remove_shard(self, shard_id: str) -> None:
        """
        Remove a shard from the hash ring.
        
        Args:
            shard_id: ID of the shard to remove
        """
        with self._lock:
            # Remove shard path
            if shard_id in self.shards:
                del self.shards[shard_id]
            
            # Remove virtual points from ring
            to_remove = []
            for hash_val, sid in self.hash_ring.items():
                if sid == shard_id:
                    to_remove.append(hash_val)
            
            for hash_val in to_remove:
                del self.hash_ring[hash_val]
            
            logger.info(f"Removed shard {shard_id}")
    
    def get_shard_for_node(self, node_id: UUID) -> Optional[str]:
        """
        Determine which shard a node belongs to.
        
        Args:
            node_id: UUID of the node
            
        Returns:
            Shard ID or None if no shards are defined
        """
        if not self.hash_ring:
            return None
        
        with self._lock:
            # Get hash value for the node
            hash_val = self._hash_key(str(node_id))
            
            # Find the next highest point on the ring
            sorted_keys = sorted(self.hash_ring.keys())
            for key in sorted_keys:
                if hash_val <= key:
                    return self.hash_ring[key]
            
            # If we went all the way around, return the first shard
            return self.hash_ring[sorted_keys[0]]
    
    def get_shard_path(self, shard_id: str) -> Optional[str]:
        """
        Get the path for a shard.
        
        Args:
            shard_id: ID of the shard
            
        Returns:
            Path to the shard data file or None if not found
        """
        return self.shards.get(shard_id)
    
    def get_all_shard_paths(self) -> List[str]:
        """Get paths to all shards."""
        return list(self.shards.values())
    
    def _hash_key(self, key: str) -> int:
        """
        Hash a key to a position on the ring.
        
        Args:
            key: String to hash
            
        Returns:
            Integer position on the hash ring
        """
        hash_obj = hashlib.md5(key.encode())
        return int(hash_obj.hexdigest(), 16)
    
    def reshard(self, new_shard_count: int) -> Dict[str, List[UUID]]:
        """
        Reorganize sharding when adding or removing shards.
        
        Args:
            new_shard_count: New number of shards
            
        Returns:
            Dictionary of shard_id -> list of node IDs that need to be moved
        """
        # This would need to be implemented based on actual node distribution
        # It would identify which nodes need to move between shards
        # For now, return an empty dict
        return {}


class DistributedStorageManager:
    """
    Manages distributed storage across multiple files or machines.
    
    Allows the merX system to scale to 100K+ records by distributing data
    across multiple storage files while maintaining the ability to
    query and retrieve data efficiently.
    """
    
    def __init__(
        self,
        base_path: str = "data/distributed",
        shard_count: int = 4,
        replicas: int = 1,
        ram_capacity_per_shard: int = 25000,
        compress: bool = True
    ):
        """
        Initialize the distributed storage manager.
        
        Args:
            base_path: Base directory for distributed storage
            shard_count: Number of shards to create
            replicas: Number of replicas per node
            ram_capacity_per_shard: Maximum nodes in RAM per shard
            compress: Whether to use compression
        """
        self.base_path = base_path
        self.shard_count = shard_count
        self.replicas = replicas
        self.ram_capacity_per_shard = ram_capacity_per_shard
        self.compress = compress
        
        # Create shard manager
        self.shard_manager = ShardManager(shard_count, replicas)
        
        # Dictionary of shard_id -> orchestrator
        self.orchestrators: Dict[str, MemoryIOOrchestrator] = {}
        
        # Global index of all nodes across shards
        self.global_index: Dict[UUID, str] = {}  # node_id -> shard_id
        
        # Lock for concurrent access
        self._lock = threading.RLock()
        
        # Global index file path
        self.global_index_path = os.path.join(base_path, "global_index.json")
        
        # Initialize
        self._init_shards()
        self._load_global_index()
        
        logger.info(f"Initialized DistributedStorageManager with {shard_count} shards at {base_path}")
    
    def _init_shards(self) -> None:
        """Initialize storage shards."""
        # Create base directory
        os.makedirs(self.base_path, exist_ok=True)
        
        # Create shards
        for i in range(self.shard_count):
            shard_id = f"shard{i:03}"
            shard_path = os.path.join(self.base_path, shard_id)
            os.makedirs(shard_path, exist_ok=True)
            
            # Create data files
            data_path = os.path.join(shard_path, "memory.mex")
            index_path = os.path.join(shard_path, "memory.mexmap")
            journal_path = os.path.join(shard_path, "journal.mexlog")
            
            # Add shard to manager
            self.shard_manager.add_shard(shard_id, shard_path)
            
            # Create orchestrator for the shard
            orchestrator = MemoryIOOrchestrator(
                data_path=data_path,
                index_path=index_path,
                journal_path=journal_path,
                ram_capacity=self.ram_capacity_per_shard
            )
            
            self.orchestrators[shard_id] = orchestrator
    
    def _load_global_index(self) -> None:
        """Load the global index of nodes to shards."""
        if os.path.exists(self.global_index_path):
            try:
                with open(self.global_index_path, 'r') as f:
                    index_data = json.load(f)
                    self.global_index = {UUID(k): v for k, v in index_data.items()}
                logger.info(f"Loaded global index with {len(self.global_index)} entries")
            except Exception as e:
                logger.error(f"Failed to load global index: {e}")
                self.global_index = {}
    
    def _save_global_index(self) -> None:
        """Save the global index of nodes to shards."""
        try:
            with open(self.global_index_path, 'w') as f:
                # Convert UUIDs to strings for JSON
                index_data = {str(k): v for k, v in self.global_index.items()}
                json.dump(index_data, f, indent=2)
            logger.debug(f"Saved global index with {len(self.global_index)} entries")
        except Exception as e:
            logger.error(f"Failed to save global index: {e}")
    
    def insert_node(self, node: Union[MemoryNode, RAMXNode]) -> UUID:
        """
        Insert a node into the distributed storage.
        
        Args:
            node: The node to insert
            
        Returns:
            UUID of the inserted node
        """
        with self._lock:
            # Determine which shard to use
            shard_id = self.shard_manager.get_shard_for_node(node.id)
            if not shard_id:
                # If no shards are defined yet, use the first one
                shard_id = f"shard000"
            
            # Get the orchestrator for this shard
            orchestrator = self.orchestrators.get(shard_id)
            if not orchestrator:
                logger.error(f"No orchestrator found for shard {shard_id}")
                # Create a new orchestrator for this shard if needed
                shard_path = os.path.join(self.base_path, shard_id)
                os.makedirs(shard_path, exist_ok=True)
                data_path = os.path.join(shard_path, "memory.mex")
                index_path = os.path.join(shard_path, "memory.mexmap")
                journal_path = os.path.join(shard_path, "journal.mexlog")
                orchestrator = MemoryIOOrchestrator(
                    data_path=data_path,
                    index_path=index_path,
                    journal_path=journal_path,
                    ram_capacity=self.ram_capacity_per_shard
                )
                self.orchestrators[shard_id] = orchestrator
            
            # Insert node into the shard
            node_id = orchestrator.insert_node(node)
            
            # Update global index
            self.global_index[node_id] = shard_id
            
            # Save global index periodically
            # In a real system, this would be more efficient
            if len(self.global_index) % 100 == 0:
                self._save_global_index()
            
            return node_id
    
    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        """
        Retrieve a node from distributed storage.
        
        Args:
            node_id: UUID of the node to retrieve
            
        Returns:
            The node if found, None otherwise
        """
        # Check if we know which shard this node is in
        shard_id = self.global_index.get(node_id)
        
        if shard_id:
            # If we know the shard, go directly there
            orchestrator = self.orchestrators.get(shard_id)
            if orchestrator:
                return orchestrator.get_node(node_id)
        
        # If we don't know the shard or didn't find it, search all shards
        for shard_id, orchestrator in self.orchestrators.items():
            node = orchestrator.get_node(node_id)
            if node:
                # Update global index for future lookups
                with self._lock:
                    self.global_index[node_id] = shard_id
                return node
        
        return None
    
    def update_node(self, node: Union[MemoryNode, RAMXNode]) -> bool:
        """
        Update an existing node in distributed storage.
        
        Args:
            node: The node to update
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we know which shard this node is in
        shard_id = self.global_index.get(node.id)
        
        if shard_id:
            # If we know the shard, go directly there
            orchestrator = self.orchestrators.get(shard_id)
            if orchestrator:
                return orchestrator.update_node(node)
        
        # If we don't know which shard or the update failed, try inserting instead
        self.insert_node(node)
        return True
    
    def get_all_nodes(self) -> List[MemoryNode]:
        """
        Get all nodes in distributed storage.
        Warning: This can be very expensive with large datasets.
        
        Returns:
            List of all memory nodes
        """
        all_nodes = []
        
        for orchestrator in self.orchestrators.values():
            # Get nodes from RAMX (in memory) for efficiency
            ramx_nodes = orchestrator.ramx.get_all_nodes()
            all_nodes.extend([n.to_memory_node() for n in ramx_nodes])
        
        return all_nodes
    
    def search_by_content(self, query: str, limit: int = 10) -> List[Tuple[MemoryNode, float]]:
        """
        Search for nodes by content across all shards.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of (node, score) tuples
        """
        all_results = []
        
        # Search each shard
        for orchestrator in self.orchestrators.values():
            # Use RAMX's neural triggering
            results = orchestrator.ramx.trigger_based_recall(query, limit * 2)
            all_results.extend([(node.to_memory_node(), score) for node, score in results])
        
        # Sort by score and limit results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:limit]
    
    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryNode]:
        """
        Search for nodes by tags across all shards.
        
        Args:
            tags: List of tags to search for
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        all_results = []
        
        # Search each shard
        for orchestrator in self.orchestrators.values():
            results = orchestrator.ramx.recall_by_tags(tags, limit * 2)
            all_results.extend([node.to_memory_node() for node in results])
        
        # Sort by activation and limit results
        all_results.sort(key=lambda x: x.activation, reverse=True)
        return all_results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the distributed storage system."""
        stats = {
            "shard_count": self.shard_count,
            "total_nodes": 0,
            "active_nodes": 0,
            "total_ram_usage_mb": 0,
            "shards": {}
        }
        
        # Collect stats from each shard
        for shard_id, orchestrator in self.orchestrators.items():
            shard_stats = orchestrator.get_stats()
            stats["total_nodes"] += shard_stats.get("total_nodes", 0)
            stats["active_nodes"] += shard_stats.get("active_nodes", 0)
            stats["total_ram_usage_mb"] += shard_stats.get("memory_usage_mb", 0)
            
            # Store individual shard stats
            stats["shards"][shard_id] = {
                "nodes": shard_stats.get("total_nodes", 0),
                "active_nodes": shard_stats.get("active_nodes", 0),
                "ram_usage_mb": shard_stats.get("memory_usage_mb", 0),
                "disk_size_mb": shard_stats.get("disk_size_bytes", 0) / (1024 * 1024)
            }
        
        return stats
    
    def flush(self) -> bool:
        """Flush all pending writes to disk."""
        success = True
        
        for orchestrator in self.orchestrators.values():
            if not orchestrator.flush():
                success = False
        
        # Also save global index
        self._save_global_index()
        
        return success
    
    def shutdown(self) -> None:
        """Shutdown the distributed storage system cleanly."""
        # Flush everything to disk
        self.flush()
        
        # Shutdown each orchestrator
        for orchestrator in self.orchestrators.values():
            orchestrator.shutdown()
        
        logger.info("Distributed storage system shutdown complete")
