"""
Memory Compression Utilities for merX.

Provides compression capabilities to further optimize memory storage
for large datasets (100K+ records).
"""

import zlib
import logging
import pickle
from typing import Any, Dict, Optional, Tuple
from uuid import UUID
import lz4.frame
import blosc
from dataclasses import asdict, is_dataclass

from src.interfaces import MemoryNode

logger = logging.getLogger(__name__)

# Compression level settings
DEFAULT_COMPRESSION_LEVEL = 5  # Balance between speed and size
MAX_COMPRESSION_LEVEL = 9  # Maximum compression (slowest)
FAST_COMPRESSION_LEVEL = 1  # Fast compression (less efficient)


class MemoryCompression:
    """
    Memory compression utilities for efficient storage of large datasets.

    Implements multiple compression algorithms that can be selected based on
    performance requirements:
    - zlib: Good general-purpose compression
    - lz4: Extremely fast compression/decompression
    - blosc: High-performance scientific data compression
    """

    @staticmethod
    def compress_node(
        node: MemoryNode,
        algorithm: str = "zlib",
        level: int = DEFAULT_COMPRESSION_LEVEL,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress a memory node using the specified algorithm.

        Args:
            node: The memory node to compress
            algorithm: Compression algorithm ('zlib', 'lz4', or 'blosc')
            level: Compression level (1-9, higher = more compression but slower)

        Returns:
            Tuple of (compressed_data, metadata)
        """
        try:
            # Convert node to dict for serialization
            if is_dataclass(node):
                node_dict = asdict(node)
            else:
                node_dict = {
                    "id": str(node.id),
                    "content": node.content,
                    "tags": list(node.tags),
                    "links": [(str(link.to_id), link.weight) for link in node.links],
                    "timestamp": node.timestamp.isoformat(),
                    "activation": node.activation,
                    "node_type": node.node_type,
                    "version": node.version,
                    "version_of": str(node.version_of) if node.version_of else None,
                }

            # Serialize to bytes
            serialized_data = pickle.dumps(node_dict)
            original_size = len(serialized_data)

            # Compress based on algorithm
            if algorithm == "zlib":
                compressed_data = zlib.compress(serialized_data, level)
            elif algorithm == "lz4":
                compressed_data = lz4.frame.compress(
                    serialized_data, compression_level=level
                )
            elif algorithm == "blosc":
                compressed_data = blosc.compress(serialized_data, clevel=level)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")

            compressed_size = len(compressed_data)
            compression_ratio = (
                original_size / compressed_size if compressed_size > 0 else 1.0
            )

            metadata = {
                "algorithm": algorithm,
                "level": level,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "node_id": str(node.id),
            }

            logger.debug(
                f"Compressed node {node.id}: {original_size} -> {compressed_size} bytes "
                f"(ratio: {compression_ratio:.2f}x)"
            )

            return compressed_data, metadata

        except Exception as e:
            logger.error(f"Failed to compress node {node.id}: {e}")
            raise

    @staticmethod
    def decompress_node(
        compressed_data: bytes, metadata: Dict[str, Any]
    ) -> Optional[MemoryNode]:
        """
        Decompress memory node data back to a MemoryNode object.

        Args:
            compressed_data: The compressed binary data
            metadata: Metadata containing compression details

        Returns:
            Reconstructed MemoryNode or None if decompression fails
        """
        try:
            algorithm = metadata.get("algorithm", "zlib")

            # Decompress based on algorithm
            if algorithm == "zlib":
                serialized_data = zlib.decompress(compressed_data)
            elif algorithm == "lz4":
                serialized_data = lz4.frame.decompress(compressed_data)
            elif algorithm == "blosc":
                serialized_data = blosc.decompress(compressed_data)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")

            # Deserialize to dict
            node_dict = pickle.loads(serialized_data)

            # Reconstruct MemoryNode
            from datetime import datetime
            from src.interfaces import MemoryLink

            links = [
                MemoryLink(to_id=UUID(link[0]), weight=link[1])
                for link in node_dict.get("links", [])
            ]

            node = MemoryNode(
                id=UUID(node_dict["id"]),
                content=node_dict["content"],
                tags=set(node_dict["tags"]),
                links=links,
                timestamp=datetime.fromisoformat(node_dict["timestamp"]),
                activation=node_dict.get("activation", 1.0),
                node_type=node_dict.get("node_type", "note"),
                version=node_dict.get("version", 1),
                version_of=(
                    UUID(node_dict["version_of"])
                    if node_dict.get("version_of")
                    else None
                ),
            )

            logger.debug(f"Decompressed node {node.id} successfully")
            return node

        except Exception as e:
            logger.error(f"Failed to decompress node data: {e}")
            return None

    @staticmethod
    def get_optimal_algorithm(node_size: int) -> Tuple[str, int]:
        """
        Get the optimal compression algorithm and level for a given node size.

        Args:
            node_size: Size of the node data in bytes

        Returns:
            Tuple of (algorithm, compression_level)
        """
        if node_size < 1024:  # Small nodes (< 1KB)
            # For small nodes, fast compression is better
            return "lz4", FAST_COMPRESSION_LEVEL
        elif node_size < 10240:  # Medium nodes (1-10KB)
            # Balanced compression for medium nodes
            return "zlib", DEFAULT_COMPRESSION_LEVEL
        else:  # Large nodes (> 10KB)
            # High compression for large nodes
            return "blosc", MAX_COMPRESSION_LEVEL

    @staticmethod
    def estimate_memory_savings(
        nodes: Dict[UUID, MemoryNode], sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Estimate potential memory savings from compression.

        Args:
            nodes: Dictionary of nodes to analyze
            sample_size: Number of nodes to sample for estimation

        Returns:
            Dictionary with compression analysis results
        """
        if not nodes:
            return {"error": "No nodes provided for analysis"}

        # Sample nodes for analysis
        node_list = list(nodes.values())
        sample_nodes = node_list[: min(sample_size, len(node_list))]

        results = {
            "total_nodes": len(nodes),
            "sample_size": len(sample_nodes),
            "algorithms": {},
            "recommendations": {},
        }

        total_original_size = 0

        for algorithm in ["zlib", "lz4", "blosc"]:
            algorithm_stats = {
                "total_original": 0,
                "total_compressed": 0,
                "compression_ratios": [],
                "processing_times": [],
            }

            for node in sample_nodes:
                try:
                    import time

                    start_time = time.time()

                    compressed_data, metadata = MemoryCompression.compress_node(
                        node, algorithm, DEFAULT_COMPRESSION_LEVEL
                    )

                    processing_time = time.time() - start_time

                    algorithm_stats["total_original"] += metadata["original_size"]
                    algorithm_stats["total_compressed"] += metadata["compressed_size"]
                    algorithm_stats["compression_ratios"].append(
                        metadata["compression_ratio"]
                    )
                    algorithm_stats["processing_times"].append(processing_time)

                    if algorithm == "zlib":  # Only count once
                        total_original_size += metadata["original_size"]

                except Exception as e:
                    logger.warning(f"Failed to test compression with {algorithm}: {e}")

            # Calculate averages
            if algorithm_stats["compression_ratios"]:
                algorithm_stats["avg_compression_ratio"] = sum(
                    algorithm_stats["compression_ratios"]
                ) / len(algorithm_stats["compression_ratios"])
                algorithm_stats["avg_processing_time"] = sum(
                    algorithm_stats["processing_times"]
                ) / len(algorithm_stats["processing_times"])
                algorithm_stats["total_savings_percent"] = (
                    (
                        (
                            algorithm_stats["total_original"]
                            - algorithm_stats["total_compressed"]
                        )
                        / algorithm_stats["total_original"]
                    )
                    * 100
                    if algorithm_stats["total_original"] > 0
                    else 0
                )

            results["algorithms"][algorithm] = algorithm_stats

        # Generate recommendations
        if results["algorithms"]:
            best_ratio = max(
                results["algorithms"].values(),
                key=lambda x: x.get("avg_compression_ratio", 0),
            )
            fastest = min(
                results["algorithms"].values(),
                key=lambda x: x.get("avg_processing_time", float("inf")),
            )

            results["recommendations"] = {
                "best_compression": next(
                    k for k, v in results["algorithms"].items() if v == best_ratio
                ),
                "fastest": next(
                    k for k, v in results["algorithms"].items() if v == fastest
                ),
                "estimated_total_savings_gb": (
                    (total_original_size * len(nodes) / len(sample_nodes)) / (1024**3)
                    if sample_nodes
                    else 0
                ),
            }

        return results
