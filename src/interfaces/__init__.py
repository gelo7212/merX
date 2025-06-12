"""
Core interfaces for the merX memory system.
Defines protocols for all major components to ensure loose coupling.
"""

from typing import Protocol, Dict, List, Optional, Any, BinaryIO
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryLink:
    """Represents a weighted link between memory nodes."""

    to_id: UUID
    weight: float
    link_type: str = "default"


@dataclass
class MemoryNode:
    """Core memory node structure."""

    id: UUID
    content: str
    node_type: str
    version: int
    timestamp: datetime
    activation: float
    decay_rate: float
    version_of: Optional[UUID] = None
    links: List[MemoryLink] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.tags is None:
            self.tags = []


class IMemorySerializer(Protocol):
    """Interface for memory node serialization/deserialization."""

    def write_node(self, file: BinaryIO, node: MemoryNode) -> int:
        """Write a memory node to binary file. Returns bytes written."""
        ...

    def read_node(self, file: BinaryIO) -> MemoryNode:
        """Read a memory node from binary file."""
        ...

    def calculate_node_size(self, node: MemoryNode) -> int:
        """Calculate the size in bytes that a node would occupy."""
        ...


class IIndexManager(Protocol):
    """Interface for managing UUID to byte offset mappings."""

    def load_index(self, path: str) -> Dict[UUID, int]:
        """Load the index from .mexmap file."""
        ...

    def save_index(self, path: str, index: Dict[UUID, int]) -> None:
        """Save the index to .mexmap file."""
        ...

    def update_index(self, path: str, node_id: UUID, offset: int) -> None:
        """Update a single entry in the index."""
        ...


class IMemoryStorage(Protocol):
    """Interface for low-level memory storage operations."""

    def append_node(self, node: MemoryNode) -> int:
        """Append a node to storage. Returns byte offset."""
        ...

    def read_node(self, offset: int) -> MemoryNode:
        """Read a node at the given byte offset."""
        ...

    def read_node_by_id(self, node_id: UUID) -> Optional[MemoryNode]:
        """Read a node by its UUID."""
        ...

    def get_all_nodes(self) -> List[MemoryNode]:
        """Get all nodes (for maintenance operations)."""
        ...


class IDecayProcessor(Protocol):
    """Interface for memory decay operations."""

    def apply_decay(self, node: MemoryNode, current_time: datetime) -> MemoryNode:
        """Apply decay to a memory node based on time elapsed."""
        ...

    def calculate_decay(self, node: MemoryNode, current_time: datetime) -> float:
        """Calculate the decay amount without applying it."""
        ...

    def refresh_activation(self, node: MemoryNode, boost: float = 0.1) -> MemoryNode:
        """Refresh/boost activation when memory is accessed."""
        ...


class IMemoryLinker(Protocol):
    """Interface for managing links between memory nodes."""

    def create_link(
        self, from_id: UUID, to_id: UUID, weight: float, link_type: str = "default"
    ) -> None:
        """Create a link between two memory nodes."""
        ...

    def update_link_weight(self, from_id: UUID, to_id: UUID, new_weight: float) -> None:
        """Update the weight of an existing link."""
        ...

    def get_linked_nodes(
        self, node_id: UUID, min_weight: float = 0.0
    ) -> List[MemoryLink]:
        """Get all nodes linked from the given node."""
        ...

    def find_path(
        self, from_id: UUID, to_id: UUID, max_depth: int = 3
    ) -> Optional[List[UUID]]:
        """Find a path between two nodes through links."""
        ...


class IRecallEngine(Protocol):
    """Interface for memory recall and activation spreading."""

    def recall_by_content(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """Recall memories by content similarity."""
        ...

    def recall_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryNode]:
        """Recall memories by tag matching."""
        ...

    def spreading_activation(
        self, start_nodes: List[UUID], max_depth: int = 3
    ) -> Dict[UUID, float]:
        """Perform spreading activation from starting nodes."""
        ...

    def get_top_activated(self, limit: int = 10) -> List[MemoryNode]:
        """Get the most activated memory nodes."""
        ...


class IVersionManager(Protocol):
    """Interface for memory versioning operations."""

    def create_version(
        self, original_id: UUID, new_content: str, **kwargs
    ) -> MemoryNode:
        """Create a new version of an existing memory."""
        ...

    def get_version_chain(self, node_id: UUID) -> List[MemoryNode]:
        """Get the complete version chain for a memory."""
        ...

    def resolve_latest_version(self, node_id: UUID) -> MemoryNode:
        """Get the most recent version of a memory."""
        ...

    def resolve_original_version(self, node_id: UUID) -> MemoryNode:
        """Get the original version of a memory."""
        ...


class IMemoryEngine(Protocol):
    """Main interface for the memory engine - orchestrates all operations."""

    def insert_memory(
        self,
        content: str,
        node_type: str = "fact",
        tags: List[str] = None,
        related_ids: List[UUID] = None,
        version_of: Optional[UUID] = None,
    ) -> UUID:
        """Insert a new memory node."""
        ...

    def get_memory(self, node_id: UUID) -> Optional[MemoryNode]:
        """Retrieve a memory by ID."""
        ...

    def update_memory_activation(self, node_id: UUID, boost: float = 0.1) -> None:
        """Update activation when memory is accessed."""
        ...

    def recall_memories(
        self, query: str = None, tags: List[str] = None, limit: int = 10
    ) -> List[MemoryNode]:
        """Recall memories based on query and/or tags."""
        ...

    def apply_global_decay(self) -> int:
        """Apply decay to all memories. Returns number of nodes processed."""
        ...

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        ...
