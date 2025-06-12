# 🧠 merX Memory System

A neural-inspired memory graph system with binary storage format (.mex) for efficient, append-only memory operations.

## Features

- **Binary Memory Nodes**: Custom .mex format for fast serialization
- **Graph-based Memory**: Weighted links between memory nodes
- **Versioning**: Immutable memory evolution with version tracking
- **Decay System**: Neural-like activation decay over time
- **Spreading Activation**: Brain-inspired recall mechanisms
- **Dependency Injection**: Clean, testable architecture

## Quick Start

```python
from src.container.di_container import create_container
from src.engine.memory_engine import MemoryEngine

# Initialize the system
container = create_container()
engine = container.memory_engine()

# Insert memories
node_id = engine.insert_memory("Ana loves coffee", tags=["personal", "preference"])
related_id = engine.insert_memory("Coffee shops in downtown", related_ids=[node_id])

# Recall memories
memories = engine.recall_memories("coffee", limit=5)
```

## Architecture

The system follows SOLID principles with:
- Interface segregation via Python protocols
- Dependency injection for loose coupling
- Separation of concerns across modules
- Immutable memory operations

## File Format

See the [.mex specification](docs/mex_format.md) for details on the binary format.
