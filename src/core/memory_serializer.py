"""
Binary serialization for memory nodes in .mex format.
Implements the custom binary format specification.
"""

import struct
import io
from typing import BinaryIO
from uuid import UUID
from datetime import datetime
import logging

from src.interfaces import IMemorySerializer, MemoryNode, MemoryLink

logger = logging.getLogger(__name__)


class MemorySerializer(IMemorySerializer):
    """
    Handles binary serialization/deserialization of memory nodes.
    
    Binary format per node:
    - Header: 4 bytes 'MEX0'
    - Node ID: 36 bytes (UUID string)
    - Version: 2 bytes (uint16)
    - Timestamp: 8 bytes (double)
    - Type: 1 byte length + variable UTF-8
    - Content Length: 4 bytes (uint32)
    - Content: variable UTF-8
    - Link Count: 2 bytes (uint16)
    - Links: [36 bytes UUID + 4 bytes weight] * count
    - Activation: 4 bytes (float32)
    - Decay Rate: 4 bytes (float32)
    - Version Of: 1 byte flag + optional 36 bytes UUID
    - Tag Count: 1 byte
    - Tags: [1 byte length + variable UTF-8] * count
    """
    
    HEADER_MARKER = b'MEX0'
    ENCODING = 'utf-8'
    
    def serialize_to_bytes(self, node: MemoryNode) -> bytes:
        """Serialize a node to bytes without writing to a file."""
        import io
        buffer = io.BytesIO()
        self.write_node(buffer, node)
        return buffer.getvalue()
    
    def write_node(self, file: BinaryIO, node: MemoryNode) -> int:
        """Write a memory node to binary file."""
        start_pos = file.tell()
        
        try:
            # Header marker
            file.write(self.HEADER_MARKER)
            
            # Node ID (36 bytes UUID string)
            uuid_bytes = str(node.id).encode(self.ENCODING)
            if len(uuid_bytes) != 36:
                raise ValueError(f"Invalid UUID length: {len(uuid_bytes)}")
            file.write(uuid_bytes)
            
            # Version (uint16)
            file.write(struct.pack('<H', node.version))
            
            # Timestamp (double - Unix timestamp)
            timestamp = node.timestamp.timestamp()
            file.write(struct.pack('<d', timestamp))
            
            # Node type (length prefixed string)
            type_bytes = node.node_type.encode(self.ENCODING)
            file.write(struct.pack('<B', len(type_bytes)))
            file.write(type_bytes)
            
            # Content (length prefixed string)
            content_bytes = node.content.encode(self.ENCODING)
            file.write(struct.pack('<I', len(content_bytes)))
            file.write(content_bytes)
              # Links
            links = node.links if node.links is not None else []
            file.write(struct.pack('<H', len(links)))
            for link in links:
                link_uuid_bytes = str(link.to_id).encode(self.ENCODING)
                file.write(link_uuid_bytes)
                file.write(struct.pack('<f', link.weight))
                # Link type (length prefixed)
                link_type_bytes = link.link_type.encode(self.ENCODING)
                file.write(struct.pack('<B', len(link_type_bytes)))
                file.write(link_type_bytes)
            
            # Activation and decay rate
            file.write(struct.pack('<f', node.activation))
            file.write(struct.pack('<f', node.decay_rate))
            
            # Version of (optional)
            if node.version_of:
                file.write(struct.pack('<B', 1))  # Has version_of
                version_of_bytes = str(node.version_of).encode(self.ENCODING)
                file.write(version_of_bytes)
            else:
                file.write(struct.pack('<B', 0))  # No version_of
            
            # Tags
            tags = node.tags if node.tags is not None else []
            file.write(struct.pack('<B', len(tags)))
            for tag in tags:
                tag_bytes = tag.encode(self.ENCODING)
                file.write(struct.pack('<B', len(tag_bytes)))
                file.write(tag_bytes)
            
            bytes_written = file.tell() - start_pos
            logger.debug(f"Wrote node {node.id} ({bytes_written} bytes)")
            return bytes_written
            
        except Exception as e:
            logger.error(f"Failed to write node {node.id}: {e}")
            raise
    
    def read_node(self, file: BinaryIO) -> MemoryNode:
        """Read a memory node from binary file."""
        start_pos = file.tell()
        
        try:
            # Read and verify header
            header = file.read(4)
            if header != self.HEADER_MARKER:
                raise ValueError(f"Invalid header marker: {header}")
            
            # Node ID
            uuid_bytes = file.read(36)
            node_id = UUID(uuid_bytes.decode(self.ENCODING))
            
            # Version
            version = struct.unpack('<H', file.read(2))[0]
            
            # Timestamp
            timestamp_float = struct.unpack('<d', file.read(8))[0]
            timestamp = datetime.fromtimestamp(timestamp_float)
            
            # Node type
            type_len = struct.unpack('<B', file.read(1))[0]
            node_type = file.read(type_len).decode(self.ENCODING)
            
            # Content
            content_len = struct.unpack('<I', file.read(4))[0]
            content = file.read(content_len).decode(self.ENCODING)
            
            # Links
            link_count = struct.unpack('<H', file.read(2))[0]
            links = []
            for _ in range(link_count):
                link_uuid_bytes = file.read(36)
                link_to_id = UUID(link_uuid_bytes.decode(self.ENCODING))
                link_weight = struct.unpack('<f', file.read(4))[0]
                # Link type
                link_type_len = struct.unpack('<B', file.read(1))[0]
                link_type = file.read(link_type_len).decode(self.ENCODING)
                
                links.append(MemoryLink(
                    to_id=link_to_id,
                    weight=link_weight,
                    link_type=link_type
                ))
            
            # Activation and decay rate
            activation = struct.unpack('<f', file.read(4))[0]
            decay_rate = struct.unpack('<f', file.read(4))[0]
            
            # Version of (optional)
            has_version_of = struct.unpack('<B', file.read(1))[0]
            version_of = None
            if has_version_of:
                version_of_bytes = file.read(36)
                version_of = UUID(version_of_bytes.decode(self.ENCODING))
            
            # Tags
            tag_count = struct.unpack('<B', file.read(1))[0]
            tags = []
            for _ in range(tag_count):
                tag_len = struct.unpack('<B', file.read(1))[0]
                tag = file.read(tag_len).decode(self.ENCODING)
                tags.append(tag)
            
            node = MemoryNode(
                id=node_id,
                content=content,
                node_type=node_type,
                version=version,
                timestamp=timestamp,
                activation=activation,
                decay_rate=decay_rate,
                version_of=version_of,
                links=links,
                tags=tags
            )
            
            bytes_read = file.tell() - start_pos
            logger.debug(f"Read node {node.id} ({bytes_read} bytes)")
            return node
            
        except Exception as e:
            logger.error(f"Failed to read node at position {start_pos}: {e}")
            raise
    
    def calculate_node_size(self, node: MemoryNode) -> int:
        """Calculate the size in bytes that a node would occupy."""
        size = 0
        
        # Header + UUID + version + timestamp
        size += 4 + 36 + 2 + 8
        
        # Type (length + content)
        size += 1 + len(node.node_type.encode(self.ENCODING))
        
        # Content (length + content)
        size += 4 + len(node.content.encode(self.ENCODING))
          # Links (count + links)
        size += 2  # link count
        links = node.links if node.links is not None else []
        for link in links:
            size += 36 + 4  # UUID + weight
            size += 1 + len(link.link_type.encode(self.ENCODING))  # link type
        
        # Activation + decay rate
        size += 4 + 4
        
        # Version of (flag + optional UUID)
        size += 1
        if node.version_of:
            size += 36
        
        # Tags (count + tags)
        size += 1  # tag count
        tags = node.tags if node.tags is not None else []
        for tag in tags:
            size += 1 + len(tag.encode(self.ENCODING))
        
        return size
