"""
Enhanced Memory Engine Factory - Creates and configures enhanced memory engines.

This factory simplifies the creation of memory engines with enhanced components
like RAMX and Memory I/O Orchestrator for high-performance operation.
"""

import os
from typing import Optional
from uuid import UUID

from src.container.enhanced_di_container import create_enhanced_container, EnhancedContainer
from src.engine.memory_engine import MemoryEngine
from src.core.memory_io_orchestrator import MemoryIOOrchestrator


class EnhancedMemoryEngineFactory:
    """
    Factory for creating enhanced memory engines with optimized components.
    
    This factory simplifies the process of creating and configuring memory engines
    with high-performance RAM-first architecture.
    """
    
    @staticmethod
    def create_engine(
        config_path: Optional[str] = None,
        ram_capacity: int = 100000,
        data_path: str = "data/memory.mex",
        flush_interval: float = 5.0,
        flush_threshold: int = 100
    ) -> MemoryEngine:
        """
        Create an enhanced memory engine with optimized components.
        
        Args:
            config_path: Optional path to configuration file
            ram_capacity: Maximum number of nodes to keep in RAM
            data_path: Path to memory data file
            flush_interval: Seconds between auto-flushes to disk
            flush_threshold: Number of writes before auto-flush
            
        Returns:
            Configured MemoryEngine instance with enhanced components
        """
        # Create container
        container = create_enhanced_container(config_path)
        
        # Apply custom settings
        container.config.storage.ram_capacity.override(ram_capacity)
        container.config.storage.data_path.override(data_path)
        container.config.storage.index_path.override(data_path.replace('.mex', '.mexmap'))
        container.config.storage.journal_path.override(data_path.replace('.mex', '.mexlog'))
        container.config.storage.flush_interval.override(flush_interval)
        container.config.storage.flush_threshold.override(flush_threshold)
        
        # Get memory engine
        return container.memory_engine()
    
    @staticmethod
    def get_container_with_engine() -> tuple[EnhancedContainer, MemoryEngine]:
        """
        Get both the container and engine for advanced configuration.
        
        Returns:
            Tuple of (container, engine)
        """
        container = create_enhanced_container()
        engine = container.memory_engine()
        return container, engine
    
    @staticmethod
    def cleanup_and_shutdown(engine: MemoryEngine) -> None:
        """
        Properly shut down the memory engine and flush data to disk.
        
        Args:
            engine: The memory engine to shut down
        """
        # Check if storage is Memory I/O Orchestrator
        if hasattr(engine.storage, 'flush') and callable(engine.storage.flush):
            # Cast to Memory I/O Orchestrator
            orchestrator: MemoryIOOrchestrator = engine.storage
            orchestrator.flush()
            orchestrator.shutdown()
        else:
            # Standard shutdown
            pass  # Could add standard cleanup here if needed
