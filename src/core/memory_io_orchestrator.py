"""
Memory I/O Orchestrator - Coordinates between RAM and disk storage.
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

from src.interfaces import MemoryNode, IMemoryStorage
from src.core.ramx import RAMX, RAMXNode
from src.core.index_manager import IndexManager
from src.core.memory_serializer import MemorySerializer

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """Entry in the write-ahead journal."""

    node_id: UUID
    operation: str  # 'insert', 'update', 'delete'
    timestamp: float
    serialized_data: bytes


class MemoryIOOrchestrator:
    """
    Coordinates memory operations between RAM and disk storage.

    Features:
    - RAM-first access pattern for speed
    - Write-behind for durability
    - Batch disk operations for performance
    - Journal for crash recovery
    """

    def __init__(
        self,
        data_path: str = "data/memory.mex",
        index_path: str = "data/memory.mexmap",
        journal_path: str = "data/journal.mexlog",
        flush_interval: float = 5.0,  # seconds between flushes
        flush_threshold: int = 100,  # flush after this many writes
        ram_capacity: int = 100000,  # max nodes in RAM
    ):
        """
        Initialize the memory I/O orchestrator.

        Args:
            data_path: Path to memory data file (.mex)
            index_path: Path to index file (.mexmap)
            journal_path: Path to journal file (.mexlog)
            flush_interval: Seconds between auto-flushes
            flush_threshold: Number of writes before auto-flush
            ram_capacity: Maximum number of nodes to keep in RAM
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Core components
        self.data_path = data_path
        self.index_path = index_path
        self.journal_path = journal_path

        # RAM memory store
        self.ramx = RAMX(capacity=ram_capacity)

        # Enhanced index manager
        self.index_manager = IndexManager()

        # Memory serializer for binary encoding
        self.serializer = MemorySerializer()

        # Write queue for batching
        self._write_queue = queue.Queue()
        self._pending_writes = 0

        # Control parameters
        self.flush_interval = flush_interval
        self.flush_threshold = flush_threshold

        # Thread control
        self._shutdown_flag = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()        # Load existing data into RAM
        self._load_existing_data()
        
        logger.info(
            "Initialized Memory I/O Orchestrator with paths: %s, %s", 
            data_path, index_path
        )

    def insert_node(self, node: Union[MemoryNode, RAMXNode]) -> UUID:
        """
        Insert a new memory node.

        Args:
            node: The node to insert

        Returns:
            UUID of the inserted node
        """
        # Convert to RAMXNode if needed
        if isinstance(node, MemoryNode):
            ramx_node = RAMXNode.from_memory_node(node)
        else:
            ramx_node = node

        # Add to RAM immediately
        self.ramx.add_node(ramx_node)

        # Queue for disk write
        self._queue_write(ramx_node, "insert")

        logger.debug("Inserted node %s into RAM and queued for disk", ramx_node.id)
        return ramx_node.id

    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: The UUID of the node to retrieve

        Returns:
            The node if found, None otherwise
        """
        # Try RAM first (fast path)
        ramx_node = self.ramx.get_node(node_id)

        if ramx_node:
            # Convert to MemoryNode interface
            return ramx_node.to_memory_node()

        # If not in RAM, try loading from disk
        try:
            # Check if we have the offset in the index
            offset = self.index_manager.get_offset(node_id)
            if offset is None:
                # Load the entire index in case it wasn't in memory
                index = self.index_manager.load_index(self.index_path)
                offset = index.get(node_id)

            if offset is not None:
                # Read from disk at the specific offset
                with open(self.data_path, "rb") as f:
                    f.seek(offset)
                    node = self.serializer.read_node(f)

                    # Add to RAM cache for future access
                    self.ramx.add_memory_node(node)

                    return node
        except Exception as e:
            logger.error("Failed to retrieve node %s from disk: %s", node_id, e)

        return None

    def update_node(self, node: Union[MemoryNode, RAMXNode]) -> bool:
        """
        Update an existing memory node.
        Note: In .mex format, nodes are append-only, so this creates a new version.

        Args:
            node: The updated node

        Returns:
            True if successful, False otherwise
        """
        # Convert to RAMXNode if needed
        if isinstance(node, MemoryNode):
            ramx_node = RAMXNode.from_memory_node(node)
        else:
            ramx_node = node

        # Check if the node exists
        existing = self.ramx.get_node(ramx_node.id)

        if not existing:
            # Check disk
            disk_node = self.get_node(ramx_node.id)
            if not disk_node:
                # Node doesn't exist anywhere
                return False

        # Update in RAM
        self.ramx.add_or_update_node(ramx_node)

        # Queue for disk write
        self._queue_write(ramx_node, "update")

        logger.debug("Updated node %s in RAM and queued for disk", ramx_node.id)
        return True

    def flush(self) -> bool:
        """
        Explicitly flush pending writes to disk.

        Returns:
            True if successful, False otherwise
        """
        if self._pending_writes == 0:
            return True  # Nothing to flush

        try:
            # Process all pending writes
            success = self._process_write_queue()

            # Flush the index manager's batch updates
            if success:
                self.index_manager.flush_batch_updates(self.index_path)

            return success
        except Exception as e:
            logger.error(f"Failed to flush pending writes: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the orchestrator, ensuring all data is persisted."""
        # Signal the flush thread to stop
        self._shutdown_flag.set()

        # Process any remaining writes
        self.flush()

        # Wait for flush thread to finish
        self._flush_thread.join(timeout=5.0)

        # Shutdown RAMX
        self.ramx.shutdown()

        logger.info("Memory I/O Orchestrator shutdown complete")

    def search_by_content(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """
        Search for nodes by content.

        Args:
            query: The search query
            limit: Maximum number of results

        Returns:
            List of matching nodes
        """
        # Use RAMX's neural triggering
        results = self.ramx.trigger_based_recall(query, limit)

        # Convert to MemoryNode interface
        return [node.to_memory_node() for node, _ in results]

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryNode]:
        """
        Search for nodes by tags.

        Args:
            tags: List of tags to search for
            limit: Maximum number of results

        Returns:
            List of matching nodes
        """
        # Use RAMX's tag search
        results = self.ramx.recall_by_tags(tags, limit)

        # Convert to MemoryNode interface
        return [node.to_memory_node() for node in results]
        
    def get_all_nodes(self) -> List[MemoryNode]:
        """
        Get all nodes from storage.
        
        Returns:
            List of all memory nodes
        """
        # Get all nodes from RAM
        ramx_nodes = self.ramx.get_all_nodes()
        
        # Convert to MemoryNode interface
        return [node.to_memory_node() for node in ramx_nodes]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        ram_stats = self.ramx.get_stats()

        # Add disk stats
        try:
            disk_size = (
                os.path.getsize(self.data_path) if os.path.exists(self.data_path) else 0
            )
            index_size = (
                os.path.getsize(self.index_path)
                if os.path.exists(self.index_path)
                else 0
            )
        except Exception:
            disk_size = 0
            index_size = 0

        return {
            **ram_stats,
            "disk_size_bytes": disk_size,
            "index_size_bytes": index_size,
            "pending_writes": self._pending_writes,
        }

    def _queue_write(self, node: RAMXNode, operation: str) -> None:
        """Queue a write operation."""
        try:
            # Serialize the node
            node_data = self.serializer.serialize_to_bytes(node.to_memory_node())

            # Create journal entry
            entry = JournalEntry(
                node_id=node.id,
                operation=operation,
                timestamp=time.time(),
                serialized_data=node_data,
            )

            # Add to queue
            self._write_queue.put(entry)
            self._pending_writes += 1

            # Write to journal for crash recovery
            self._write_to_journal(entry)

            # Auto-flush if threshold reached
            if self._pending_writes >= self.flush_threshold:
                self.flush()

        except Exception as e:
            logger.error(f"Failed to queue write for node {node.id}: {e}")

    def _write_to_journal(self, entry: JournalEntry) -> None:
        """Write an entry to the journal file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)

            # Write entry to journal
            with open(self.journal_path, "ab") as f:
                # Write a fixed-length header
                header = f"{entry.node_id}:{entry.operation}:{entry.timestamp}:{len(entry.serialized_data)}".encode()
                f.write(header.ljust(128, b" "))

                # Write the actual data
                f.write(entry.serialized_data)

                # Ensure data is on disk
                f.flush()
                os.fsync(f.fileno())

        except Exception as e:
            logger.error(f"Failed to write to journal: {e}")

    def _flush_worker(self) -> None:
        """Background thread for periodic flushing."""
        while not self._shutdown_flag.is_set():
            try:
                # Sleep for the flush interval
                time.sleep(self.flush_interval)

                # If there are pending writes, flush them
                if self._pending_writes > 0:
                    self.flush()

            except Exception as e:
                logger.error(f"Error in flush worker: {e}")

    def _process_write_queue(self) -> bool:
        """Process pending writes from the queue."""
        if self._pending_writes == 0:
            return True

        try:
            # Open data file for append
            with open(self.data_path, "ab") as f:
                # Process all entries in the queue
                processed_count = 0

                while (
                    not self._write_queue.empty() and processed_count < 1000
                ):  # Limit batch size
                    try:
                        # Get next entry
                        entry = self._write_queue.get_nowait()

                        # Get current position in file
                        offset = f.tell()

                        # Write data
                        f.write(entry.serialized_data)

                        # Update index
                        self.index_manager.update_index(
                            self.index_path, entry.node_id, offset
                        )

                        # Mark as done
                        self._write_queue.task_done()
                        processed_count += 1

                    except queue.Empty:
                        break

                # Ensure data is on disk
                f.flush()
                os.fsync(f.fileno())

                # Update pending count
                self._pending_writes -= processed_count

                logger.debug(f"Flushed {processed_count} writes to disk")
                return True

        except Exception as e:
            logger.error(f"Failed to process write queue: {e}")
            return False

    def _load_existing_data(self) -> None:
        """Load existing data from disk into RAM on startup."""
        try:
            # Check if data file exists
            if not os.path.exists(self.data_path):
                logger.info(f"No existing data file found at {self.data_path}")
                return

            # Load the index
            index = self.index_manager.load_index(self.index_path)
            if not index:
                logger.warning("Index is empty, cannot load data efficiently")
                return

            logger.info(f"Loading {len(index)} nodes from disk into RAM...")

            # Load a sample of nodes into RAM (most recent first)
            sample_size = min(len(index), self.ramx.get_stats().get("capacity", 100000))
            sorted_offsets = sorted(index.items(), key=lambda x: x[1], reverse=True)
            sample_offsets = sorted_offsets[:sample_size]

            # Open data file
            with open(self.data_path, "rb") as f:
                for node_id, offset in sample_offsets:
                    try:
                        # Seek to offset
                        f.seek(offset)

                        # Read node
                        node = self.serializer.read_node(f)

                        # Add to RAM
                        self.ramx.add_memory_node(node)

                    except Exception as e:
                        logger.error(
                            f"Failed to load node {node_id} at offset {offset}: {e}"
                        )

            logger.info(f"Loaded {sample_size} nodes into RAM")

        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")

    def read_node_by_id(self, node_id: UUID) -> Optional[MemoryNode]:
        """
        Read a node from storage by its ID.

        Args:
            node_id: The UUID of the node to retrieve

        Returns:
            The node if found, None otherwise
        """
        # Check RAM first (fast path)
        ramx_node = self.ramx.get_node(node_id)
        if ramx_node:
            return ramx_node.to_memory_node()

        # If not found in RAM, try disk (slow path)
        # In a real implementation, we would check the index and read from disk
        # For now, just return None as not found
        logger.debug(f"Node {node_id} not found in RAM or disk")
        return None

    def serialize_to_bytes(self, node: MemoryNode) -> bytes:
        """
        Serialize a node to bytes.

        Args:
            node: The node to serialize

        Returns:
            Serialized node as bytes
        """
        return self.serializer.serialize_to_bytes(node)

    def append_node(self, node: MemoryNode) -> Optional[UUID]:
        """Append a memory node to storage."""
        return self.insert_node(node)
