"""
Enhanced index manager with RAM caching and file locking for the merX memory system.
Cross-platform version that works on both Windows and Linux.
"""

import os
import json
import time
import logging
import random
import threading
from typing import Dict, Optional
from uuid import UUID

import portalocker  # For cross-platform file locking

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Enhanced index manager with RAM caching, file locking, and retry logic.

    Features:
    - In-memory index cache for fast lookups
    - File locking to prevent concurrent write issues
    - Retry logic for resilience
    - Batch updates for performance
    """

    def __init__(self, max_retries: int = 5, retry_delay: float = 0.1):
        """
        Initialize the enhanced index manager.

        Args:
            max_retries: Maximum number of retries for file operations
            retry_delay: Base delay between retries (will use exponential backoff)
        """
        self._index_cache: Dict[UUID, int] = {}
        self._index_lock = threading.RLock()
        self._batch_updates: Dict[UUID, int] = {}
        self._batch_lock = threading.RLock()
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._batch_size = 100  # Number of updates before forcing a write

        logger.info("Initialized enhanced index manager")

    def load_index(self, path: str) -> Dict[UUID, int]:
        """Load the index from .mexmap file with retry logic."""
        with self._index_lock:
            # If we already have the index in memory, return it
            if self._index_cache:
                logger.debug(
                    f"Using cached index with {len(self._index_cache)} entries"
                )
                return self._index_cache.copy()

            for attempt in range(self._max_retries):
                try:
                    # Check if the file exists
                    if not os.path.exists(path):
                        logger.info(
                            f"Index file {path} does not exist, creating empty index"
                        )
                        self._index_cache = {}
                        return {}

                    # Open the file with shared read lock
                    with portalocker.Lock(path, "r", timeout=10) as f:
                        data = json.load(f)

                    # Convert string keys back to UUIDs
                    self._index_cache = {UUID(k): v for k, v in data.items()}
                    logger.info(
                        f"Loaded index with {len(self._index_cache)} entries from {path}"
                    )
                    return self._index_cache.copy()

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid index file format: {e}")
                    return {}
                except Exception as e:
                    if attempt < self._max_retries - 1:
                        delay = (
                            self._retry_delay * (2**attempt) * (0.5 + random.random())
                        )
                        logger.warning(
                            f"Failed to load index (attempt {attempt+1}/{self._max_retries}): {e}, retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to load index after {self._max_retries} attempts: {e}"
                        )
                        return {}

    def save_index(self, path: str, index: Dict[UUID, int]) -> bool:
        """Save the index to .mexmap file with retry and file locking."""
        with self._index_lock:
            for attempt in range(self._max_retries):
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    # Convert UUIDs to strings for JSON serialization
                    data = {str(k): v for k, v in index.items()}

                    # Write atomically using a temporary file
                    temp_path = f"{path}.{os.getpid()}.tmp"

                    # Write to temp file with exclusive lock
                    with portalocker.Lock(temp_path, "w", timeout=10) as f:
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk

                    # Atomic rename with retries
                    rename_success = self._safe_rename(temp_path, path)

                    if rename_success:
                        # Update memory cache
                        self._index_cache = index.copy()
                        logger.info(f"Saved index with {len(index)} entries to {path}")
                        return True
                    else:
                        # If rename failed after retries, report error
                        logger.error(
                            f"Failed to rename temp file to {path} after {self._max_retries} attempts"
                        )
                        return False

                except Exception as e:
                    if attempt < self._max_retries - 1:
                        delay = (
                            self._retry_delay * (2**attempt) * (0.5 + random.random())
                        )
                        logger.warning(
                            f"Failed to save index (attempt {attempt+1}/{self._max_retries}): {e}, retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to save index after {self._max_retries} attempts: {e}"
                        )
                        # Clean up temp file if it exists
                        temp_path = f"{path}.{os.getpid()}.tmp"
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception as clean_e:
                                logger.warning(
                                    f"Failed to clean up temp file {temp_path}: {clean_e}"
                                )
                        return False

    def update_index(self, path: str, node_id: UUID, offset: int) -> bool:
        """
        Update a single entry in the index with batching support.

        Args:
            path: Path to .mexmap file
            node_id: UUID of the node
            offset: Byte offset in the .mex file

        Returns:
            True if update was successful or queued, False if failed
        """
        with self._batch_lock:
            # Update the batch
            self._batch_updates[node_id] = offset

            # Update the in-memory cache immediately for fast lookup
            with self._index_lock:
                self._index_cache[node_id] = offset

            # If batch is large enough, commit it
            if len(self._batch_updates) >= self._batch_size:
                return self.flush_batch_updates(path)
            return True

    def flush_batch_updates(self, path: str) -> bool:
        """
        Flush all batched updates to disk.

        Args:
            path: Path to .mexmap file

        Returns:
            True if successful, False if failed
        """
        with self._batch_lock:
            if not self._batch_updates:
                return True  # Nothing to flush

            # Load the current index
            current_index = self.load_index(path)

            # Apply all pending updates
            for node_id, offset in self._batch_updates.items():
                current_index[node_id] = offset

            # Clear batch regardless of save result
            batch_size = len(self._batch_updates)
            self._batch_updates.clear()

            # Save the updated index
            success = self.save_index(path, current_index)
            if success:
                logger.info(f"Flushed {batch_size} batch updates to {path}")
            else:
                logger.error(f"Failed to flush {batch_size} batch updates to {path}")

            return success

    def get_offset(self, node_id: UUID) -> Optional[int]:
        """Get the byte offset for a node ID from the in-memory cache."""
        with self._index_lock:
            return self._index_cache.get(node_id)
            
    def batch_update_index(self, path: str, updates: Dict[UUID, int]) -> bool:
        """
        Update multiple index entries at once.

        Args:
            path: Path to .mexmap file
            updates: Dictionary mapping node IDs to offsets

        Returns:
            True if successful, False if failed
        """
        with self._batch_lock:
            # Update the batch with all entries
            self._batch_updates.update(updates)

            # Update the in-memory cache immediately for fast lookup
            with self._index_lock:
                for node_id, offset in updates.items():
                    self._index_cache[node_id] = offset

            # If batch is large enough, commit it
            if len(self._batch_updates) >= self._batch_size:
                return self.flush_batch_updates(path)
            return True
            
    def _safe_rename(self, src: str, dst: str) -> bool:
        """Safely rename a file with retries."""
        for attempt in range(self._max_retries):
            try:
                # On Windows, we may need to handle the case where the target file exists
                if os.path.exists(dst):
                    os.replace(src, dst)  # Overwrite existing file
                else:
                    os.rename(src, dst)  # Simple rename
                return True
            except Exception as e:
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt) * (0.5 + random.random())
                    logger.warning(
                        f"Rename failed (attempt {attempt+1}/{self._max_retries}): {e}, retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Rename failed after {self._max_retries} attempts: {e}"
                    )
                    return False
        
        # Fallback return in case loop exits unexpectedly
        return False
