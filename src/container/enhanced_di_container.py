"""
Enhanced dependency injection container for the merX memory system.
Uses RAMX and other enhanced components for high-performance operation.

Features:
- RAM-first architecture with Memory I/O Orchestrator
- Enhanced recall engine with optimized algorithms
- Distributed storage support for 100K+ records
- Memory compression for efficient storage
"""

import os
import logging
from typing import Optional, Dict, Any
from dependency_injector import containers, providers
import structlog

from src.core.memory_serializer import MemorySerializer
from src.core.index_manager import IndexManager
from src.core.memory_io_orchestrator import MemoryIOOrchestrator
from src.core.decay_processor import DecayProcessor
from src.core.memory_linker import MemoryLinker
from src.engine.recall_engine import RecallEngine
from src.engine.version_manager import VersionManager
from src.engine.memory_engine import MemoryEngine
from src.core.ramx import RAMX
from src.utils.compression import MemoryCompression
from src.adapters.memory_storage_adapter import MemoryStorageAdapter


class EnhancedContainer(containers.DeclarativeContainer):
    """
    Enhanced dependency injection container with RAMX and Memory I/O Orchestrator.

    This container configures and wires all components with proper dependencies
    using the high-performance components.
    """

    # Configuration
    config = providers.Configuration()

    # Logging
    logger = providers.Singleton(structlog.get_logger)

    # Core components - use Singleton for shared resources
    memory_serializer = providers.Singleton(MemorySerializer)

    # Enhanced components
    enhanced_index_manager = providers.Singleton(IndexManager)

    # RAMX for in-memory storage
    ramx = providers.Singleton(
        RAMX,
        capacity=config.storage.ram_capacity,
        activation_threshold=config.recall.activation_threshold,
        spreading_decay=config.recall.spreading_decay,
        max_hops=config.recall.max_hops,
    )

    # Memory I/O Orchestrator
    memory_io_orchestrator = providers.Singleton(
        MemoryIOOrchestrator,
        data_path=config.storage.data_path,
        index_path=config.storage.index_path,
        journal_path=config.storage.journal_path,
        flush_interval=config.storage.flush_interval,
        flush_threshold=config.storage.flush_threshold,
        ram_capacity=config.storage.ram_capacity,
    )

    decay_processor = providers.Factory(
        DecayProcessor,
        decay_model=config.decay.model,
        min_activation=config.decay.min_activation,
    )

    memory_linker = providers.Factory(
        MemoryLinker,
        storage=memory_io_orchestrator,  # Use orchestrator instead of direct storage
    )
    # Storage adapter for compatibility
    memory_storage_adapter = providers.Factory(
        MemoryStorageAdapter, orchestrator=memory_io_orchestrator
    )

    # Memory compression utilities
    memory_compression = providers.Factory(
        MemoryCompression
    )  # Engine components - Ultimate Enhanced recall engine for maximum performance
    recall_engine = providers.Factory(
        RecallEngine,
        memory_storage=memory_io_orchestrator,  # Use orchestrator instead of direct storage
    )

    # Legacy recall engine as fallback
    standard_recall_engine = providers.Factory(
        RecallEngine,
        memory_storage=memory_io_orchestrator,
    )

    version_manager = providers.Factory(
        VersionManager,
        storage=memory_io_orchestrator,  # Use orchestrator instead of direct storage
    )

    # Main memory engine
    memory_engine = providers.Factory(
        MemoryEngine,
        storage=memory_io_orchestrator,  # Use orchestrator instead of direct storage
        decay_processor=decay_processor,
        linker=memory_linker,
        recall_engine=recall_engine,
        version_manager=version_manager,
    )


def create_enhanced_container(config_path: Optional[str] = None) -> EnhancedContainer:
    """Create a container with enhanced components for high-performance operation."""
    container = EnhancedContainer()

    # Default config
    container.config.from_dict(
        {
            "storage": {
                "data_path": "data/memory.mex",
                "index_path": "data/memory.mexmap",
                "journal_path": "data/journal.mexlog",
                "flush_interval": 5.0,  # seconds
                "flush_threshold": 100,  # write operations
                "ram_capacity": 100000,  # nodes
            },
            "decay": {
                "model": "exponential",
                "min_activation": 0.01,
                "half_life": 30,  # days
            },
            "recall": {
                "activation_threshold": 0.01,
                "spreading_decay": 0.7,
                "max_hops": 3,
            },
            "performance": {
                "max_workers": 4,  # Number of threads for parallel operations
                "use_cache": True,  # Enable result caching
                "cache_size": 1000,  # Maximum cache entries
                "compress_data": True,  # Use compression
                "compression_algorithm": "zlib",  # zlib, lz4, or blosc
                "compression_level": 5,  # 1-9 (higher = better compression but slower)
            },
            "distributed": {
                "enabled": False,  # Enable distributed storage
                "shard_count": 4,  # Number of shards
                "shard_replicas": 1,  # Number of replicas per shard
                "sharding_strategy": "consistent",  # Sharding strategy
            },
        }
    )

    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        container.config.from_ini(config_path)

    return container


def create_enhanced_test_container() -> EnhancedContainer:
    """Create a container configured for testing with enhanced components."""
    container = EnhancedContainer()

    # Test config with in-memory paths
    container.config.from_dict(
        {
            "storage": {
                "data_path": "data/test_memory.mex",
                "index_path": "data/test_memory.mexmap",
                "journal_path": "data/test_journal.mexlog",
                "flush_interval": 1.0,  # faster flush for tests
                "flush_threshold": 10,  # smaller batch size for tests
                "ram_capacity": 100000,  # smaller RAM capacity for tests
            },
            "decay": {
                "model": "exponential",
                "min_activation": 0.01,
                "half_life": 1,  # faster decay for tests
            },
            "recall": {
                "activation_threshold": 0.01,
                "spreading_decay": 0.7,
                "max_hops": 3,
            },
            "performance": {
                "max_workers": 2,  # Reduced workers for tests
                "use_cache": True,
                "cache_size": 100,  # Smaller cache for tests
                "compress_data": False,  # No compression in tests for speed
                "compression_algorithm": "zlib",
                "compression_level": 1,
            },
            "distributed": {
                "enabled": False,
                "shard_count": 2,
                "shard_replicas": 1,
                "sharding_strategy": "consistent",
            },
        }
    )

    return container
