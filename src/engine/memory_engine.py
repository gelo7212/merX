"""
Main memory engine - orchestrates all memory operations.
"""

from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import logging
import re

from src.interfaces import (
    IMemoryEngine,
    IMemoryStorage,
    IDecayProcessor,
    IMemoryLinker,
    IRecallEngine,
    IVersionManager,
    MemoryNode,
    MemoryLink,
)

logger = logging.getLogger(__name__)


class MemoryEngine(IMemoryEngine):
    """
    Main memory engine that orchestrates all memory operations.

    Provides high-level interface for:
    - Inserting and retrieving memories
    - Managing activation and decay
    - Performing recall operations
    - Handling versioning and linking
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        decay_processor: IDecayProcessor,
        linker: IMemoryLinker,
        recall_engine: IRecallEngine,
        version_manager: IVersionManager,
    ):
        """
        Initialize memory engine with all required components.

        Args:
            storage: Memory storage interface
            decay_processor: Handles memory decay
            linker: Manages memory links
            recall_engine: Handles memory recall
            version_manager: Manages memory versioning
        """
        self.storage = storage
        self.decay_processor = decay_processor
        self.linker = linker
        self.recall_engine = recall_engine
        self.version_manager = version_manager

        logger.info("Initialized memory engine")

    def insert_memory(
        self,
        content: str,
        node_type: str = "fact",
        tags: Optional[List[str]] = None,
        related_ids: Optional[List[UUID]] = None,
        version_of: Optional[UUID] = None,
        decay_rate: float = 0.02,
    ) -> UUID:
        """Insert a new memory node."""
        if not content.strip():
            raise ValueError("Memory content cannot be empty")

        # Extract tags if not provided
        if tags is None:
            tags = self._extract_tags(content)

        # Prepare links list
        links = []

        # Add links to related memories
        if related_ids:
            for related_id in related_ids:
                # Verify the related node exists
                if self.storage.read_node_by_id(related_id):
                    links.append(
                        MemoryLink(to_id=related_id, weight=0.6, link_type="related")
                    )
                else:
                    logger.warning(
                        f"Related node {related_id} not found, skipping link"
                    )

        # If this is a version, create strong link to original
        if version_of:
            if self.storage.read_node_by_id(version_of):
                links.append(
                    MemoryLink(to_id=version_of, weight=0.9, link_type="version")
                )
            else:
                logger.warning(f"Version parent {version_of} not found")

        # Create new memory node with all links
        node = MemoryNode(
            id=uuid4(),
            content=content.strip(),
            node_type=node_type,
            version=1 if version_of is None else self._get_next_version(version_of),
            timestamp=datetime.now(),
            activation=1.0,  # New memories start with full activation
            decay_rate=decay_rate,
            version_of=version_of,
            links=links,
            tags=tags or [],
        )

        # Store the node with all links included
        self.storage.append_node(node)

        logger.info(
            f"Inserted memory {node.id}: '{content[:50]}...' (type={node_type})"
        )
        return node.id

    def get_memory(self, node_id: UUID) -> Optional[MemoryNode]:
        """Retrieve a memory by ID."""
        node = self.storage.read_node_by_id(node_id)

        if node:
            # Apply decay and refresh activation
            current_time = datetime.now()
            node = self.decay_processor.apply_decay(node, current_time)
            node = self.decay_processor.refresh_activation(node, boost=0.05)

            logger.debug(
                f"Retrieved memory {node_id} (activation={node.activation:.3f})"
            )

        return node

    def update_memory_activation(self, node_id: UUID, boost: float = 0.1) -> None:
        """Update activation when memory is accessed."""
        node = self.storage.read_node_by_id(node_id)
        if not node:
            logger.warning(f"Cannot update activation for non-existent node {node_id}")
            return

        # Store original activation for logging
        original_activation = node.activation

        # Refresh activation
        updated_node = self.decay_processor.refresh_activation(node, boost)

        # Try multiple update methods for different storage types
        update_success = False        # Method 1: Direct RAMX update (fastest for RAMX-based storage)
        if hasattr(self.storage, 'ramx') and self.storage.ramx:  # type: ignore
            try:
                ramx_node = self.storage.ramx.get_node(node_id)  # type: ignore
                if ramx_node:
                    ramx_node.boost(boost)
                    update_success = True
                    logger.debug(
                        f"Updated RAMX activation for {node_id}: {original_activation:.3f} -> {ramx_node.activation:.3f}"
                    )
            except Exception as e:
                logger.debug(f"RAMX update failed for {node_id}: {e}")

        # Method 2: Storage update_node method
        if not update_success and hasattr(self.storage, 'update_node'):
            try:
                success = self.storage.update_node(updated_node)  # type: ignore
                if success:
                    update_success = True
                    logger.debug(
                        f"Updated storage activation for {node_id}: {original_activation:.3f} -> {updated_node.activation:.3f}"
                    )
            except Exception as e:
                logger.debug(f"Storage update failed for {node_id}: {e}")

        # Method 3: Specific activation update method
        if not update_success and hasattr(self.storage, 'update_node_activation'):
            try:
                success = self.storage.update_node_activation(node_id, updated_node.activation)  # type: ignore
                if success:
                    update_success = True
                    logger.debug(
                        f"Updated node activation for {node_id}: {original_activation:.3f} -> {updated_node.activation:.3f}"
                    )
            except Exception as e:
                logger.debug(f"Activation update failed for {node_id}: {e}")

        # Log result
        if not update_success:
            logger.debug(
                f"Activation update not persisted for {node_id}: {original_activation:.3f} -> {updated_node.activation:.3f}"
            )

    def recall_memories(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        context_ids: Optional[List[UUID]] = None,
    ) -> List[MemoryNode]:
        """Recall memories based on query and/or tags."""
        memories = []

        # Use content-based recall if query provided
        if query:
            memories = self.recall_engine.recall_by_content(query, limit)

        # Use tag-based recall if tags provided
        elif tags:
            memories = self.recall_engine.recall_by_tags(tags, limit)

        # Fall back to top activated memories
        else:
            memories = self.recall_engine.get_top_activated(limit)        # Update activation for recalled memories
        for memory in memories:
            self.update_memory_activation(memory.id, boost=0.03)
        
        logger.debug(f"Recalled {len(memories)} memories")
        return memories

    def apply_global_decay(self) -> int:
        """Apply decay to all memories. Returns number of nodes processed."""
        processed_count = 0
        current_time = datetime.now()        # Try RAMX direct decay first (most efficient)
        if hasattr(self.storage, 'ramx') and self.storage.ramx:  # type: ignore
            try:
                processed_count = self.storage.ramx.apply_global_decay()  # type: ignore
                logger.info(f"Applied RAMX decay to {processed_count} memory nodes")
                return processed_count
            except Exception as e:
                logger.warning(f"RAMX decay failed, falling back to storage decay: {e}")

        # Try storage-level decay
        if hasattr(self.storage, 'apply_global_decay'):
            try:
                processed_count = self.storage.apply_global_decay()  # type: ignore
                logger.info(f"Applied storage decay to {processed_count} memory nodes")
                return processed_count
            except Exception as e:
                logger.warning(f"Storage decay failed, falling back to manual decay: {e}")

        # Fallback: manual decay processing
        all_nodes = self.storage.get_all_nodes()
        
        for node in all_nodes:
            try:
                # Apply decay
                decayed_node = self.decay_processor.apply_decay(node, current_time)

                # Try multiple update methods
                update_success = False

                # Method 1: Direct RAMX update
                if hasattr(self.storage, 'ramx') and self.storage.ramx:  # type: ignore
                    try:
                        ramx_node = self.storage.ramx.get_node(node.id)  # type: ignore
                        if ramx_node:
                            ramx_node.decay(current_time.timestamp())
                            update_success = True
                    except Exception as e:
                        logger.debug(f"RAMX decay update failed for {node.id}: {e}")

                # Method 2: Storage update
                if not update_success and hasattr(self.storage, 'update_node'):
                    try:
                        self.storage.update_node(decayed_node)  # type: ignore
                        update_success = True
                    except Exception as e:
                        logger.debug(f"Storage update failed for {node.id}: {e}")

                if update_success:
                    processed_count += 1

            except Exception as e:
                logger.error(f"Failed to apply decay to {node.id}: {e}")

        logger.info(f"Applied manual decay to {processed_count} memory nodes")
        return processed_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        all_nodes = self.storage.get_all_nodes()

        if not all_nodes:
            return {
                "total_memories": 0,
                "active_memories": 0,
                "average_activation": 0.0,
                "memory_types": {},
                "total_tags": 0,
                "total_links": 0,
            }

        # Basic stats
        total_memories = len(all_nodes)
        active_memories = len([n for n in all_nodes if n.activation > 0.1])
        avg_activation = sum(n.activation for n in all_nodes) / total_memories

        # Memory types
        type_counts = {}
        for node in all_nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1        # Tags
        all_tags = set()
        for node in all_nodes:
            if node.tags:  # Check if tags is not None/empty
                all_tags.update(node.tags)

        # Links
        total_links = sum(len(node.links) if node.links else 0 for node in all_nodes)

        return {
            "total_memories": total_memories,
            "active_memories": active_memories,
            "average_activation": avg_activation,
            "memory_types": type_counts,
            "total_tags": len(all_tags),
            "total_links": total_links,
        }

    def create_memory_version(
        self, original_id: UUID, new_content: str, **kwargs
    ) -> UUID:
        """Create a new version of an existing memory."""
        new_node = self.version_manager.create_version(
            original_id, new_content, **kwargs
        )

        logger.info(f"Created version {new_node.id} of memory {original_id}")
        return new_node.id

    def get_memory_chain(self, node_id: UUID) -> List[MemoryNode]:
        """Get the complete version chain for a memory."""
        return self.version_manager.get_version_chain(node_id)

    def find_related_memories(
        self, node_id: UUID, max_depth: int = 2
    ) -> List[MemoryNode]:
        """Find memories related through links."""
        # Use spreading activation to find related memories
        activation_map = self.recall_engine.spreading_activation([node_id], max_depth)

        # Convert to memory nodes
        related_memories = []
        for related_id, activation in activation_map.items():
            if related_id != node_id:  # Exclude the original node
                memory = self.storage.read_node_by_id(related_id)
                if memory:
                    related_memories.append(memory)

        # Sort by activation level
        related_memories.sort(key=lambda x: x.activation, reverse=True)

        logger.debug(f"Found {len(related_memories)} related memories for {node_id}")
        return related_memories

    def maintenance(self) -> Dict[str, int]:
        """
        Perform maintenance operations on the memory system.

        Returns:
            Dictionary with counts of maintenance operations performed
        """
        logger.info("Starting memory system maintenance")

        results = {"nodes_decayed": 0}

        try:
            # Apply global decay
            results["nodes_decayed"] = self.apply_global_decay()

            logger.info(f"Maintenance completed: {results}")

        except Exception as e:
            logger.error(f"Maintenance failed: {e}")

        return results

    def _extract_tags(self, content: str) -> List[str]:
        """Simple tag extraction from content."""
        # This is a basic implementation - could be enhanced with NLP

        # Extract words that look like tags (capitalized words, etc.)
        words = re.findall(r"\b[A-Z][a-z]+\b", content)

        # Also look for hashtag-style tags
        hashtags = re.findall(r"#(\w+)", content)

        # Combine and limit to reasonable number
        tags = list(set(words + hashtags))[:10]

        return tags

    def _get_next_version(self, original_id: UUID) -> int:
        """Get the next version number for a memory."""
        try:
            chain = self.version_manager.get_version_chain(original_id)
            if chain:
                return max(node.version for node in chain) + 1
            return 2  # If original is version 1, next is 2
        except Exception:
            return 2

    def _create_automatic_link(
        self, from_id: UUID, to_id: UUID, weight: float, link_type: str = "related"
    ) -> None:
        """Create a link between memories (internal helper)."""
        try:
            self.linker.create_link(from_id, to_id, weight, link_type)
        except Exception as e:
            logger.warning(f"Failed to create automatic link {from_id} -> {to_id}: {e}")

    def link_memories(
        self, from_id: UUID, to_id: UUID, weight: float = 0.5, link_type: str = "cross_domain"
    ) -> bool:
        """
        Create a link between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            weight: Link weight (0.0 to 1.0)
            link_type: Type of link (e.g., "cross_domain", "related", "similar")

        Returns:
            True if link was created successfully, False otherwise
        """
        try:
            # Validate that both nodes exist
            from_node = self.storage.read_node_by_id(from_id)
            to_node = self.storage.read_node_by_id(to_id)

            if not from_node:
                logger.warning(f"Source node {from_id} not found")
                return False

            if not to_node:
                logger.warning(f"Target node {to_id} not found")
                return False

            # Create the link using the linker
            self.linker.create_link(from_id, to_id, weight, link_type)

            logger.info(f"Created {link_type} link: {from_id} -> {to_id} (weight={weight})")
            return True

        except Exception as e:
            logger.error(f"Failed to create link {from_id} -> {to_id}: {e}")
            return False
