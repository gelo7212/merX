"""
Low-level memory storage operations for .mex files.
"""

import os
from typing import List, Optional
from uuid import UUID
import logging

from src.interfaces import IMemoryStorage, IMemorySerializer, IIndexManager, MemoryNode

logger = logging.getLogger(__name__)


class MemoryStorage(IMemoryStorage):
    """
    Handles low-level storage operations for memory nodes.
    Manages both the .mex data file and .mexmap index file.
    """
    
    def __init__(
        self,
        data_path: str,
        serializer: IMemorySerializer,
        index_manager: IIndexManager
    ):
        """
        Initialize memory storage.
        
        Args:
            data_path: Path to the .mex file (without extension)
            serializer: Memory node serializer
            index_manager: Index manager for .mexmap file
        """
        self.data_path = data_path + '.mex'
        self.index_path = data_path + '.mexmap'
        self.serializer = serializer
        self.index_manager = index_manager
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Load index into memory for fast access
        self._index_cache = self.index_manager.load_index(self.index_path)
        
        logger.info(f"Initialized memory storage: {self.data_path}")
    
    def append_node(self, node: MemoryNode) -> int:
        """Append a node to storage. Returns byte offset."""
        try:
            # Open file in append binary mode
            with open(self.data_path, 'ab') as f:
                # Get current position (this will be our offset)
                offset = f.tell()
                
                # Write the node
                bytes_written = self.serializer.write_node(f, node)
                
                # Update index cache and persist
                self._index_cache[node.id] = offset
                self.index_manager.update_index(self.index_path, node.id, offset)
                
                logger.debug(f"Appended node {node.id} at offset {offset} ({bytes_written} bytes)")
                return offset
                
        except Exception as e:
            logger.error(f"Failed to append node {node.id}: {e}")
            raise
    
    def read_node(self, offset: int) -> MemoryNode:
        """Read a node at the given byte offset."""
        try:
            with open(self.data_path, 'rb') as f:
                f.seek(offset)
                node = self.serializer.read_node(f)
                logger.debug(f"Read node {node.id} from offset {offset}")
                return node
                
        except Exception as e:
            logger.error(f"Failed to read node at offset {offset}: {e}")
            raise
    
    def read_node_by_id(self, node_id: UUID) -> Optional[MemoryNode]:
        """Read a node by its UUID."""
        offset = self._index_cache.get(node_id)
        if offset is None:
            logger.debug(f"Node {node_id} not found in index")
            return None
        
        try:
            return self.read_node(offset)
        except Exception as e:
            logger.error(f"Failed to read node {node_id}: {e}")
            return None
    
    def get_all_nodes(self) -> List[MemoryNode]:
        """Get all nodes (for maintenance operations)."""
        nodes = []
        
        if not os.path.exists(self.data_path):
            return nodes
        
        # First try reading using the index (faster and more reliable)
        if self._index_cache:
            logger.debug(f"Reading {len(self._index_cache)} nodes using index")
            for node_id, offset in self._index_cache.items():
                try:
                    node = self.read_node(offset)
                    nodes.append(node)
                except Exception as e:
                    logger.warning(f"Failed to read indexed node {node_id} at offset {offset}: {e}")
            
            if nodes:
                logger.info(f"Loaded {len(nodes)} nodes from storage using index")
                return nodes
        
        # Fallback: scan the entire file only if index reading failed
        logger.debug("Index reading failed, scanning entire file for nodes")
        try:
            with open(self.data_path, 'rb') as f:
                while True:
                    try:
                        # Try to read a node at current position
                        position = f.tell()
                        node = self.serializer.read_node(f)
                        nodes.append(node)
                        
                        # Update index if needed
                        if node.id not in self._index_cache:
                            logger.debug(f"Found unindexed node {node.id} at position {position}")
                            self._index_cache[node.id] = position
                        
                    except EOFError:
                        # End of file reached
                        break
                    except Exception as e:
                        logger.error(f"Error reading node at position {position}: {e}")
                        # Stop on error to prevent infinite loops
                        break
            
            logger.info(f"Loaded {len(nodes)} nodes from storage")
            
            # Update index if we found new nodes
            if len(nodes) > len(self._index_cache):
                logger.info("Updating index with newly found nodes")
                self.index_manager.save_index(self.index_path, self._index_cache)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to read all nodes: {e}")
            return nodes
    
    def rebuild_index(self) -> None:
        """Rebuild the index by scanning the entire data file."""
        logger.info("Rebuilding index from data file")
        
        self._index_cache.clear()
        
        if not os.path.exists(self.data_path):
            self.index_manager.save_index(self.index_path, self._index_cache)
            return
        
        try:
            with open(self.data_path, 'rb') as f:
                while True:
                    try:
                        position = f.tell()
                        node = self.serializer.read_node(f)
                        self._index_cache[node.id] = position
                        
                    except EOFError:
                        break
                    except Exception as e:
                        logger.error(f"Error during index rebuild at position {position}: {e}")
                        break
            
            # Save the rebuilt index
            self.index_manager.save_index(self.index_path, self._index_cache)
            logger.info(f"Rebuilt index with {len(self._index_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise
