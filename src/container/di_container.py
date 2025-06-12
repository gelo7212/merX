"""
Dependency injection container for the merX memory system.
"""

import os
from typing import Optional
from dependency_injector import containers, providers
import structlog

from src.core.memory_serializer import MemorySerializer
from src.core.index_manager import IndexManager
from src.core.memory_storage import MemoryStorage
from src.core.decay_processor import DecayProcessor
from src.core.memory_linker import MemoryLinker
from src.engine.recall_engine import RecallEngine
from src.engine.version_manager import VersionManager
from src.engine.memory_engine import MemoryEngine


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for the merX memory system.
    
    Configures and wires all components with proper dependencies.
    """
    
    # Configuration
    config = providers.Configuration()
    
    # Logging
    logger = providers.Singleton(
        structlog.get_logger
    )
      # Core components - use Singleton for shared resources
    memory_serializer = providers.Singleton(
        MemorySerializer
    )
    
    index_manager = providers.Singleton(
        IndexManager
    )
    
    memory_storage = providers.Singleton(
        MemoryStorage,
        data_path=config.storage.data_path,
        serializer=memory_serializer,
        index_manager=index_manager
    )
    
    decay_processor = providers.Factory(
        DecayProcessor,
        decay_model=config.decay.model,
        min_activation=config.decay.min_activation
    )
    
    memory_linker = providers.Factory(
        MemoryLinker,
        storage=memory_storage
    )
    
    # Engine components
    recall_engine = providers.Factory(
        RecallEngine,
        storage=memory_storage,
        linker=memory_linker,
        activation_threshold=config.recall.activation_threshold,
        spreading_decay=config.recall.spreading_decay
    )
    
    version_manager = providers.Factory(
        VersionManager,
        storage=memory_storage
    )
    
    # Main memory engine
    memory_engine = providers.Factory(
        MemoryEngine,
        storage=memory_storage,
        decay_processor=decay_processor,
        linker=memory_linker,
        recall_engine=recall_engine,
        version_manager=version_manager
    )


def create_container(config_path: Optional[str] = None) -> Container:
    """
    Create and configure the dependency injection container.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Configured container instance
    """
    container = Container()
    
    # Default configuration
    default_config = {
        "storage": {
            "data_path": os.path.join(os.getcwd(), "data", "memory")
        },
        "decay": {
            "model": "exponential",
            "min_activation": 0.01
        },
        "recall": {
            "activation_threshold": 0.1,
            "spreading_decay": 0.7
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configurations (simple merge)
        for key, value in user_config.items():
            if key in default_config:
                if isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            else:
                default_config[key] = value
    
    container.config.from_dict(default_config)
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return container


def create_default_container() -> Container:
    """Create a container with default settings for quick setup."""
    return create_container()


def create_test_container() -> Container:
    """Create a container configured for testing."""
    container = Container()
    
    # Test configuration with in-memory or temporary storage
    test_config = {
        "storage": {
            "data_path": os.path.join("tests", "temp", "memory")
        },
        "decay": {
            "model": "linear",  # More predictable for testing
            "min_activation": 0.05
        },
        "recall": {
            "activation_threshold": 0.05,  # Lower threshold for testing
            "spreading_decay": 0.5
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    container.config.from_dict(test_config)
    return container
