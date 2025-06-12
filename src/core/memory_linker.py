"""
Memory linking system - manages connections between memory nodes.
"""

from typing import List, Optional, Set, Dict
from uuid import UUID
from collections import deque
import logging

from src.interfaces import IMemoryLinker, IMemoryStorage, MemoryLink

logger = logging.getLogger(__name__)


class MemoryLinker(IMemoryLinker):
    """
    Manages links between memory nodes.
    
    Provides graph operations like pathfinding, link weight management,
    and neighbor discovery for spreading activation algorithms.
    """
    
    def __init__(self, storage: IMemoryStorage):
        """
        Initialize memory linker.
        
        Args:
            storage: Memory storage for reading/writing nodes
        """
        self.storage = storage
        self._link_cache: Dict[UUID, List[MemoryLink]] = {}
        self._cache_dirty = True
        
        logger.info("Initialized memory linker")
    
    def create_link(self, from_id: UUID, to_id: UUID, weight: float, link_type: str = "default") -> None:
        """Create a link between two memory nodes."""
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"Link weight must be between 0.0 and 1.0, got {weight}")
        
        # Load the source node
        from_node = self.storage.read_node_by_id(from_id)
        if not from_node:
            raise ValueError(f"Source node {from_id} not found")
        
        # Check if target node exists
        to_node = self.storage.read_node_by_id(to_id)
        if not to_node:
            raise ValueError(f"Target node {to_id} not found")
        
        # Check if link already exists
        existing_link = None
        for link in from_node.links:
            if link.to_id == to_id and link.link_type == link_type:
                existing_link = link
                break
        
        if existing_link:
            # Update existing link weight
            existing_link.weight = weight
            logger.debug(f"Updated link {from_id} -> {to_id} weight to {weight}")
        else:
            # Add new link
            new_link = MemoryLink(to_id=to_id, weight=weight, link_type=link_type)
            from_node.links.append(new_link)
            logger.debug(f"Created link {from_id} -> {to_id} (weight={weight}, type={link_type})")
        
        # Save the updated node
        # Note: In a real implementation, this would need a way to update existing nodes
        # For now, we'll mark cache as dirty
        self._cache_dirty = True
        self._invalidate_cache()
    
    def update_link_weight(self, from_id: UUID, to_id: UUID, new_weight: float) -> None:
        """Update the weight of an existing link."""
        if new_weight < 0.0 or new_weight > 1.0:
            raise ValueError(f"Link weight must be between 0.0 and 1.0, got {new_weight}")
        
        from_node = self.storage.read_node_by_id(from_id)
        if not from_node:
            raise ValueError(f"Source node {from_id} not found")
        
        # Find and update the link
        link_found = False
        for link in from_node.links:
            if link.to_id == to_id:
                link.weight = new_weight
                link_found = True
                logger.debug(f"Updated link weight {from_id} -> {to_id}: {new_weight}")
                break
        
        if not link_found:
            raise ValueError(f"Link {from_id} -> {to_id} not found")
        
        # Mark cache as dirty
        self._cache_dirty = True
        self._invalidate_cache()
    
    def get_linked_nodes(self, node_id: UUID, min_weight: float = 0.0) -> List[MemoryLink]:
        """Get all nodes linked from the given node."""
        node = self.storage.read_node_by_id(node_id)
        if not node:
            return []
        
        # Filter by minimum weight
        filtered_links = [
            link for link in node.links 
            if link.weight >= min_weight
        ]
        
        # Sort by weight (descending)
        filtered_links.sort(key=lambda x: x.weight, reverse=True)
        
        return filtered_links
    
    def find_path(self, from_id: UUID, to_id: UUID, max_depth: int = 3) -> Optional[List[UUID]]:
        """Find a path between two nodes through links."""
        if from_id == to_id:
            return [from_id]
        
        # Breadth-first search with depth limit
        queue = deque([(from_id, [from_id])])
        visited: Set[UUID] = {from_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            # Check depth limit
            if len(path) > max_depth:
                continue
            
            # Get neighbors
            links = self.get_linked_nodes(current_id, min_weight=0.1)  # Only consider strong links
            
            for link in links:
                if link.to_id == to_id:
                    # Found target
                    return path + [link.to_id]
                
                if link.to_id not in visited:
                    visited.add(link.to_id)
                    queue.append((link.to_id, path + [link.to_id]))
        
        # No path found
        return None
    
    def get_strongly_connected_components(self, min_weight: float = 0.5) -> List[List[UUID]]:
        """
        Find strongly connected components in the memory graph.
        Useful for identifying clusters of related memories.
        """
        # This is a simplified version - a full implementation would use
        # Tarjan's or Kosaraju's algorithm
        
        all_nodes = self.storage.get_all_nodes()
        node_ids = {node.id for node in all_nodes}
        visited: Set[UUID] = set()
        components: List[List[UUID]] = []
        
        for node_id in node_ids:
            if node_id in visited:
                continue
            
            # Find component starting from this node
            component = self._find_component(node_id, min_weight, visited)
            if len(component) > 1:  # Only include actual components
                components.append(component)
        
        return components
    
    def _find_component(self, start_id: UUID, min_weight: float, visited: Set[UUID]) -> List[UUID]:
        """Find a connected component starting from a node."""
        component = []
        stack = [start_id]
        
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            
            visited.add(node_id)
            component.append(node_id)
            
            # Add neighbors with sufficient weight
            links = self.get_linked_nodes(node_id, min_weight)
            for link in links:
                if link.to_id not in visited:
                    stack.append(link.to_id)
        
        return component
    
    def get_link_statistics(self) -> Dict[str, float]:
        """Get statistics about the link graph."""
        all_nodes = self.storage.get_all_nodes()
        
        total_nodes = len(all_nodes)
        total_links = sum(len(node.links) for node in all_nodes)
        
        if total_nodes == 0:
            return {
                "total_nodes": 0,
                "total_links": 0,
                "average_links_per_node": 0.0,
                "link_density": 0.0,
                "average_link_weight": 0.0
            }
        
        # Calculate averages
        average_links = total_links / total_nodes
        
        # Link density (actual links / possible links)
        max_possible_links = total_nodes * (total_nodes - 1)
        link_density = total_links / max_possible_links if max_possible_links > 0 else 0.0
        
        # Average link weight
        all_weights = [
            link.weight 
            for node in all_nodes 
            for link in node.links
        ]
        average_weight = sum(all_weights) / len(all_weights) if all_weights else 0.0
        
        return {
            "total_nodes": total_nodes,
            "total_links": total_links,
            "average_links_per_node": average_links,
            "link_density": link_density,
            "average_link_weight": average_weight
        }
    
    def _invalidate_cache(self) -> None:
        """Invalidate the link cache."""
        self._link_cache.clear()
        self._cache_dirty = True
    
    def prune_weak_links(self, min_weight: float = 0.1) -> int:
        """
        Remove links below a certain weight threshold.
        Returns the number of links removed.
        """
        removed_count = 0
        all_nodes = self.storage.get_all_nodes()
        
        for node in all_nodes:
            original_count = len(node.links)
            node.links = [link for link in node.links if link.weight >= min_weight]
            removed_count += original_count - len(node.links)
        
        if removed_count > 0:
            logger.info(f"Pruned {removed_count} weak links (threshold: {min_weight})")
            self._invalidate_cache()
        
        return removed_count
