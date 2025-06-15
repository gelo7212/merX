"""
merX Memory System - External API Wrapper

This module provides a simplified interface for third-party applications
to integrate with the merX neural-inspired memory system.

Features:
- Simplified API for common memory operations
- Automatic configuration management
- Built-in error handling and logging
- Production-ready defaults
- Memory management utilities

Example Usage:
    from merx_api import MerXMemory

    # Initialize memory system
    memory = MerXMemory()

    # Store a memory
    memory_id = memory.store("Python is a programming language",
                           tags=["programming", "python"])

    # Search for memories
    results = memory.search("programming language", limit=5)

    # Get specific memory
    node = memory.get(memory_id)
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime
from pathlib import Path

# Add src to path for internal imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory
    from src.interfaces import MemoryNode, MemoryLink
    from src.engine.memory_engine import MemoryEngine
except ImportError as e:
    raise ImportError(f"Failed to import merX components: {e}")


class MerXMemory:
    """
    Simplified external API for the merX memory system.

    This class provides a clean, easy-to-use interface for third-party
    applications to integrate with merX without dealing with internal complexity.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        ram_capacity: int = 50000,
        auto_flush: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize the merX memory system.

        Args:
            data_path: Path to store memory data (default: ./data/merx_external.mex)
            ram_capacity: Maximum nodes to keep in RAM (default: 50000)
            auto_flush: Whether to auto-flush to disk (default: True)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Setup logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

        # Setup data path
        if data_path is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            env = "production"  # Set environment for data directory
            data_path = str(data_dir / env / "hp_mini.mex")

        self.data_path = data_path
        self.ram_capacity = ram_capacity

        # Initialize engine
        try:
            self.engine: MemoryEngine = EnhancedMemoryEngineFactory.create_engine(
                ram_capacity=ram_capacity,
                data_path=data_path,
                flush_interval=5.0 if auto_flush else 3600.0,
                flush_threshold=100 if auto_flush else 10000,
            )
            self.logger.info(f"merX Memory System initialized (data: {data_path})")
        except Exception as e:
            self.logger.error(f"Failed to initialize merX: {e}")
            raise

    def store(
        self,
        content: str,
        memory_type: str = "fact",
        tags: Optional[List[str]] = None,
        related_to: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """
        Store a new memory in the system.

        Args:
            content: The memory content to store
            memory_type: Type of memory (fact, event, concept, etc.)
            tags: List of tags for categorization
            related_to: IDs of related memories (string or list of strings)

        Returns:
            String representation of the memory ID
        """
        try:
            # Convert related_to to UUID list if provided
            related_ids = None
            if related_to:
                if isinstance(related_to, str):
                    related_to = [related_to]
                related_ids = []
                for rel_id in related_to:
                    try:
                        related_ids.append(UUID(rel_id))
                    except ValueError:
                        self.logger.warning(f"Invalid related ID format: {rel_id}")

            # Store the memory
            memory_id = self.engine.insert_memory(
                content=content,
                node_type=memory_type,
                tags=tags or [],
                related_ids=related_ids,
            )

            self.logger.debug(f"Stored memory: {memory_id}")
            return str(memory_id)

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        search_mode: str = "balanced",
    ) -> List[Dict[str, Any]]:
        """
        Search for memories in the system.

        Args:
            query: Text query to search for
            tags: Tags to filter by
            limit: Maximum number of results
            search_mode: Search strategy (fast, balanced, comprehensive)

        Returns:
            List of memory dictionaries with simplified structure
        """
        try:
            # Perform search
            results = self.engine.recall_memories(query=query, tags=tags, limit=limit)

            # Convert to simplified format
            simplified_results = []
            for node in results:
                simplified_results.append(self._node_to_dict(node))

            self.logger.debug(f"Search returned {len(simplified_results)} results")
            return simplified_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: String representation of memory ID

        Returns:
            Memory dictionary or None if not found
        """
        try:
            # Convert string to UUID
            uuid_id = UUID(memory_id)

            # Get the memory
            node = self.engine.get_memory(uuid_id)

            if node:
                return self._node_to_dict(node)
            return None

        except ValueError:
            self.logger.error(f"Invalid memory ID format: {memory_id}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {e}")
            raise

    def update_activation(self, memory_id: str, boost: float = 0.1) -> bool:
        """
        Boost the activation level of a memory (makes it more likely to be recalled).

        Args:
            memory_id: String representation of memory ID
            boost: Amount to boost activation (0.0 to 1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            uuid_id = UUID(memory_id)
            self.engine.update_memory_activation(uuid_id, boost)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update activation: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and performance metrics.

        Returns:
            Dictionary with system statistics
        """
        try:
            stats = {
                "data_path": self.data_path,
                "ram_capacity": self.ram_capacity,
                "system_info": {
                    "version": "1.0.0",
                    "architecture": "neural-inspired graph storage",
                },
            }

            # Add RAMX stats if available
            if hasattr(self.engine.storage, "ramx"):
                ramx_stats = self.engine.storage.ramx.get_stats()
                stats["memory_stats"] = ramx_stats

            # Add recall engine stats if available
            if hasattr(self.engine.recall_engine, "search_stats"):
                stats["search_stats"] = self.engine.recall_engine.search_stats

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def find_related(self, memory_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find memories related to a specific memory using spreading activation.

        Args:
            memory_id: String representation of memory ID
            max_depth: Maximum depth to search for relationships

        Returns:
            List of related memory dictionaries
        """
        try:
            uuid_id = UUID(memory_id)

            # Use spreading activation to find related memories
            activation_map = self.engine.recall_engine.spreading_activation(
                start_nodes=[uuid_id], max_depth=max_depth
            )

            # Convert to memory dictionaries
            related_memories = []
            for node_id, activation in activation_map.items():
                if node_id != uuid_id:  # Exclude the original memory
                    node = self.engine.get_memory(node_id)
                    if node:
                        memory_dict = self._node_to_dict(node)
                        memory_dict["activation_score"] = activation
                        related_memories.append(memory_dict)

            # Sort by activation score
            related_memories.sort(key=lambda x: x["activation_score"], reverse=True)

            return related_memories

        except Exception as e:
            self.logger.error(f"Failed to find related memories: {e}")
            return []

    def export_data(self, format: str = "json") -> Dict[str, Any]:
        """
        Export all memory data in a specified format.

        Args:
            format: Export format (currently only 'json' supported)

        Returns:
            Dictionary containing all memory data
        """
        try:
            if format != "json":
                raise ValueError("Only JSON format is currently supported")

            # Get all memories
            all_nodes = self.engine.storage.get_all_nodes()

            # Convert to export format
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "total_memories": len(all_nodes),
                    "data_path": self.data_path,
                },
                "memories": [self._node_to_dict(node) for node in all_nodes],
            }

            return export_data

        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            raise

    def cleanup(self):
        """
        Clean up resources and flush data to disk.
        """
        try:
            EnhancedMemoryEngineFactory.cleanup_and_shutdown(self.engine)
            self.logger.info("merX Memory System cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def _node_to_dict(self, node: MemoryNode) -> Dict[str, Any]:
        """Convert a MemoryNode to a simplified dictionary format."""
        return {
            "id": str(node.id),
            "content": node.content,
            "type": node.node_type,
            "tags": node.tags,
            "activation": node.activation,
            "timestamp": node.timestamp.isoformat(),
            "version": node.version,
            "links": [
                {
                    "to_id": str(link.to_id),
                    "weight": link.weight,
                    "type": link.link_type,
                }
                for link in (node.links or [])
            ],
        }

    def _setup_logging(self, level: str):
        """Setup logging configuration."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


class MerXMemoryError(Exception):
    """Base exception for merX memory operations."""

    pass


# Convenience functions for quick usage
def create_memory_system(data_path: Optional[str] = None, **kwargs) -> MerXMemory:
    """
    Create a new merX memory system instance.

    Args:
        data_path: Path to store memory data
        **kwargs: Additional configuration options

    Returns:
        MerXMemory instance
    """
    return MerXMemory(data_path=data_path, **kwargs)


def quick_search(
    query: str, data_path: Optional[str] = None, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform a quick search without creating a persistent memory system.

    Args:
        query: Search query
        data_path: Path to memory data file
        limit: Maximum results

    Returns:
        List of search results
    """
    memory = create_memory_system(data_path)
    try:
        return memory.search(query, limit=limit)
    finally:
        memory.cleanup()


# Version information
__version__ = "1.0.0"
__author__ = "merX Development Team"
__description__ = "External API wrapper for merX neural-inspired memory system"


if __name__ == "__main__":
    # Example usage demonstration
    print("merX Memory System - External API Demo")
    print("=" * 40)

    # Create memory system
    memory = MerXMemory(log_level="INFO")

    try:
        # Store some sample memories
        print("\n1. Storing sample memories...")
        id1 = memory.store(
            "Python is a high-level programming language",
            tags=["programming", "python", "language"],
        )
        id2 = memory.store(
            "Machine learning uses algorithms to find patterns",
            tags=["ai", "machine-learning", "algorithms"],
        )
        id3 = memory.store(
            "Neural networks are inspired by biological neurons",
            tags=["ai", "neural-networks", "biology"],
            related_to=id2,
        )

        print(f"Stored 3 memories: {id1[:8]}..., {id2[:8]}..., {id3[:8]}...")

        # Search for memories
        print("\n2. Searching for 'programming'...")
        results = memory.search("programming", limit=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['content'][:50]}... (tags: {result['tags']})")

        # Get specific memory
        print(f"\n3. Getting specific memory {id1[:8]}...")
        specific = memory.get(id1)
        if specific:
            print(f"  Content: {specific['content']}")
            print(f"  Tags: {specific['tags']}")

        # Find related memories
        print(f"\n4. Finding memories related to {id3[:8]}...")
        related = memory.find_related(id3)
        for i, rel in enumerate(related, 1):
            print(
                f"  {i}. {rel['content'][:40]}... (score: {rel['activation_score']:.3f})"
            )

        # Show stats
        print("\n5. System statistics...")
        stats = memory.get_stats()
        if "memory_stats" in stats:
            mem_stats = stats["memory_stats"]
            print(f"  Nodes in RAM: {mem_stats.get('nodes_in_ram', 'N/A')}")
            print(f"  Cache hits: {mem_stats.get('cache_hits', 'N/A')}")

    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        memory.cleanup()
        print("\n6. Cleanup completed")
