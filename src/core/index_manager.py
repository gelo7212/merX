"""
Index management for UUID to byte offset mappings (.mexmap files).
"""

import json
import os
from typing import Dict
from uuid import UUID
import logging

from src.interfaces import IIndexManager

logger = logging.getLogger(__name__)


class IndexManager(IIndexManager):
    """
    Manages the .mexmap index file that maps UUIDs to byte offsets.
    
    The index is stored as a JSON file for simplicity, but could be
    optimized to a binary format or B-tree for larger datasets.
    """
    
    def load_index(self, path: str) -> Dict[UUID, int]:
        """Load the index from .mexmap file."""
        if not os.path.exists(path):
            logger.info(f"Index file {path} does not exist, creating empty index")
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string keys back to UUIDs
            index = {UUID(k): v for k, v in data.items()}
            logger.info(f"Loaded index with {len(index)} entries from {path}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            # Return empty index rather than crashing
            return {}
    
    def save_index(self, path: str, index: Dict[UUID, int]) -> None:
        """Save the index to .mexmap file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert UUIDs to strings for JSON serialization
            data = {str(k): v for k, v in index.items()}
            
            # Write atomically by using a temporary file
            temp_path = path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            if os.path.exists(path):
                os.replace(temp_path, path)
            else:
                os.rename(temp_path, path)
            
            logger.info(f"Saved index with {len(index)} entries to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save index to {path}: {e}")
            # Clean up temp file if it exists
            temp_path = path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def update_index(self, path: str, node_id: UUID, offset: int) -> None:
        """Update a single entry in the index."""
        try:
            # Load existing index
            index = self.load_index(path)
            
            # Update entry
            index[node_id] = offset
            
            # Save updated index
            self.save_index(path, index)
            
            logger.debug(f"Updated index: {node_id} -> {offset}")
            
        except Exception as e:
            logger.error(f"Failed to update index for {node_id}: {e}")
            raise
