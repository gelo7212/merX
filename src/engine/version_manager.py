"""
Memory versioning system - manages memory evolution and version chains.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
import logging

from src.interfaces import IVersionManager, IMemoryStorage, MemoryNode

logger = logging.getLogger(__name__)


class VersionManager(IVersionManager):
    """
    Manages memory versioning and evolution.
    
    Provides functionality to:
    - Create new versions of existing memories
    - Track version chains and relationships
    - Resolve latest/original versions
    - Maintain immutable memory history
    """
    
    def __init__(self, storage: IMemoryStorage):
        """
        Initialize version manager.
        
        Args:
            storage: Memory storage interface
        """
        self.storage = storage
        logger.info("Initialized version manager")
    
    def create_version(self, original_id: UUID, new_content: str, **kwargs) -> MemoryNode:
        """
        Create a new version of an existing memory.
        
        Args:
            original_id: UUID of the memory to version
            new_content: New content for the version
            **kwargs: Additional fields to update (node_type, tags, etc.)
        
        Returns:
            The new version node
        """
        # Load the original memory
        original_node = self.storage.read_node_by_id(original_id)
        if not original_node:
            raise ValueError(f"Original node {original_id} not found")
        
        # Determine the root of the version chain
        root_id = self._find_version_root(original_id)
        
        # Create new version node
        new_node = MemoryNode(
            id=uuid4(),
            content=new_content,
            node_type=kwargs.get('node_type', original_node.node_type),
            version=original_node.version + 1,
            timestamp=datetime.now(),
            activation=kwargs.get('activation', 1.0),  # New versions start with high activation
            decay_rate=kwargs.get('decay_rate', original_node.decay_rate),
            version_of=root_id,  # Always point to root, not immediate parent
            links=kwargs.get('links', original_node.links.copy()),
            tags=kwargs.get('tags', original_node.tags.copy())
        )
        
        # Store the new version
        self.storage.append_node(new_node)
        
        logger.info(f"Created version {new_node.id} of {original_id} (v{new_node.version})")
        return new_node
    
    def get_version_chain(self, node_id: UUID) -> List[MemoryNode]:
        """
        Get the complete version chain for a memory.
        
        Returns all versions ordered from oldest to newest.
        """
        # Find the root of the version chain
        root_id = self._find_version_root(node_id)
        
        # Collect all versions that reference this root
        all_nodes = self.storage.get_all_nodes()
        version_chain = []
        
        # Add the root node
        root_node = self.storage.read_node_by_id(root_id)
        if root_node:
            version_chain.append(root_node)
        
        # Add all nodes that version the root
        for node in all_nodes:
            if node.version_of == root_id and node.id != root_id:
                version_chain.append(node)
        
        # Sort by version number
        version_chain.sort(key=lambda x: x.version)
        
        logger.debug(f"Found version chain of length {len(version_chain)} for {node_id}")
        return version_chain
    
    def resolve_latest_version(self, node_id: UUID) -> MemoryNode:
        """Get the most recent version of a memory."""
        version_chain = self.get_version_chain(node_id)
        
        if not version_chain:
            # Fallback to the node itself if no chain found
            node = self.storage.read_node_by_id(node_id)
            if node:
                return node
            raise ValueError(f"Node {node_id} not found")
        
        # Return the last (most recent) version
        latest = version_chain[-1]
        logger.debug(f"Latest version of {node_id}: {latest.id} (v{latest.version})")
        return latest
    
    def resolve_original_version(self, node_id: UUID) -> MemoryNode:
        """Get the original version of a memory."""
        root_id = self._find_version_root(node_id)
        root_node = self.storage.read_node_by_id(root_id)
        
        if not root_node:
            raise ValueError(f"Root node {root_id} not found")
        
        logger.debug(f"Original version of {node_id}: {root_node.id} (v{root_node.version})")
        return root_node
    
    def _find_version_root(self, node_id: UUID) -> UUID:
        """
        Find the root (original) node in a version chain.
        
        Follows the version_of links back to the original.
        """
        current_id = node_id
        visited = set()  # Prevent infinite loops
        
        while current_id not in visited:
            visited.add(current_id)
            
            current_node = self.storage.read_node_by_id(current_id)
            if not current_node:
                break
            
            if current_node.version_of is None:
                # This is the root
                return current_id
            
            # Move to the parent
            current_id = current_node.version_of
        
        # If we hit a loop or couldn't find root, return the starting node
        logger.warning(f"Could not find version root for {node_id}, using node itself")
        return node_id
    
    def get_version_statistics(self) -> dict:
        """Get statistics about versioning in the memory system."""
        all_nodes = self.storage.get_all_nodes()
        
        total_nodes = len(all_nodes)
        versioned_nodes = len([n for n in all_nodes if n.version_of is not None])
        root_nodes = len([n for n in all_nodes if n.version_of is None])
        
        # Find version chains
        version_chains = {}
        for node in all_nodes:
            if node.version_of is None:  # Root node
                chain_length = len(self.get_version_chain(node.id))
                version_chains[node.id] = chain_length
        
        longest_chain = max(version_chains.values()) if version_chains else 0
        avg_chain_length = sum(version_chains.values()) / len(version_chains) if version_chains else 0
        
        return {
            "total_nodes": total_nodes,
            "root_nodes": root_nodes,
            "versioned_nodes": versioned_nodes,
            "version_chains": len(version_chains),
            "longest_chain": longest_chain,
            "average_chain_length": avg_chain_length,
            "versioning_ratio": versioned_nodes / total_nodes if total_nodes > 0 else 0
        }
    
    def consolidate_versions(self, node_id: UUID, keep_latest: bool = True) -> MemoryNode:
        """
        Consolidate a version chain into a single node.
        
        Args:
            node_id: ID of any node in the version chain
            keep_latest: Whether to keep the latest version (True) or original (False)
        
        Returns:
            The consolidated node
        """
        version_chain = self.get_version_chain(node_id)
        
        if len(version_chain) <= 1:
            logger.debug(f"No consolidation needed for {node_id}")
            return version_chain[0] if version_chain else None
        
        # Choose which version to keep
        if keep_latest:
            primary_node = version_chain[-1]
        else:
            primary_node = version_chain[0]
        
        # Collect all links from all versions
        consolidated_links = []
        all_link_targets = set()
        
        for node in version_chain:
            for link in node.links:
                if link.to_id not in all_link_targets:
                    consolidated_links.append(link)
                    all_link_targets.add(link.to_id)
        
        # Collect all tags from all versions
        consolidated_tags = []
        all_tags = set()
        
        for node in version_chain:
            for tag in node.tags:
                if tag not in all_tags:
                    consolidated_tags.append(tag)
                    all_tags.add(tag)
        
        # Create consolidated node
        consolidated_node = MemoryNode(
            id=primary_node.id,
            content=primary_node.content,
            node_type=primary_node.node_type,
            version=len(version_chain),  # Set version to total chain length
            timestamp=primary_node.timestamp,
            activation=max(node.activation for node in version_chain),  # Use highest activation
            decay_rate=primary_node.decay_rate,
            version_of=None,  # Consolidated nodes are roots
            links=consolidated_links,
            tags=consolidated_tags
        )
        
        logger.info(f"Consolidated {len(version_chain)} versions into {consolidated_node.id}")
        return consolidated_node
    
    def prune_old_versions(self, max_versions: int = 5, min_activation: float = 0.1) -> int:
        """
        Prune old versions from version chains.
        
        Keeps the most recent versions and versions with high activation.
        
        Args:
            max_versions: Maximum number of versions to keep per chain
            min_activation: Minimum activation to preserve a version
        
        Returns:
            Number of versions pruned
        """
        all_nodes = self.storage.get_all_nodes()
        root_nodes = [n for n in all_nodes if n.version_of is None]
        
        pruned_count = 0
        
        for root_node in root_nodes:
            version_chain = self.get_version_chain(root_node.id)
            
            if len(version_chain) <= max_versions:
                continue
            
            # Sort by version (newest first)
            version_chain.sort(key=lambda x: x.version, reverse=True)
            
            # Keep the newest versions
            to_keep = version_chain[:max_versions]
            
            # Also keep any old versions with high activation
            for node in version_chain[max_versions:]:
                if node.activation >= min_activation:
                    to_keep.append(node)
                else:
                    pruned_count += 1
            
            # In a real implementation, would actually remove the pruned nodes
            # For now, just count them
        
        logger.info(f"Would prune {pruned_count} old versions")
        return pruned_count
