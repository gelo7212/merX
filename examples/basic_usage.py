"""
Basic usage example for the merX memory system.
"""

import logging
from uuid import UUID
from src.container.di_container import create_default_container

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic merX memory operations."""
    print("🧠 merX Memory System - Basic Example")
    print("=" * 50)
    
    # Initialize the memory system
    print("Initializing memory system...")
    container = create_default_container()
    engine = container.memory_engine()
    
    # Insert some memories
    print("\n📝 Inserting memories...")
    
    # Insert a fact about coffee
    coffee_id = engine.insert_memory(
        content="Coffee is a brewed drink prepared from roasted coffee beans.",
        node_type="fact",
        tags=["coffee", "beverage", "drink"]
    )
    print(f"Inserted coffee fact: {coffee_id}")
    
    # Insert a personal preference
    preference_id = engine.insert_memory(
        content="I prefer dark roast coffee in the morning.",
        node_type="preference",
        tags=["coffee", "personal", "morning"],
        related_ids=[coffee_id]  # Link to coffee fact
    )
    print(f"Inserted preference: {preference_id}")
    
    # Insert a memory about a coffee shop
    shop_id = engine.insert_memory(
        content="Blue Bottle Coffee has excellent single-origin beans.",
        node_type="experience",
        tags=["coffee", "shop", "blue bottle"],
        related_ids=[coffee_id, preference_id]
    )
    print(f"Inserted shop experience: {shop_id}")
    
    # Create a version of the preference
    updated_preference_id = engine.create_memory_version(
        preference_id,
        "I actually prefer medium roast coffee with a hint of chocolate notes.",
        tags=["coffee", "personal", "morning", "chocolate"]
    )
    print(f"Created updated preference: {updated_preference_id}")
    
    # Recall memories
    print("\n🔍 Recalling memories...")
    
    # Search for coffee-related memories
    coffee_memories = engine.recall_memories(query="coffee", limit=5)
    print(f"\nFound {len(coffee_memories)} coffee-related memories:")
    for i, memory in enumerate(coffee_memories, 1):
        print(f"  {i}. [{memory.node_type}] {memory.content[:50]}...")
        print(f"     Activation: {memory.activation:.3f}, Tags: {memory.tags}")
    
    # Search by tags
    personal_memories = engine.recall_memories(tags=["personal"], limit=3)
    print(f"\nFound {len(personal_memories)} personal memories:")
    for i, memory in enumerate(personal_memories, 1):
        print(f"  {i}. {memory.content[:50]}...")
    
    # Find related memories using spreading activation
    print(f"\n🌐 Finding memories related to coffee fact...")
    related_memories = engine.find_related_memories(coffee_id, max_depth=2)
    print(f"Found {len(related_memories)} related memories:")
    for i, memory in enumerate(related_memories, 1):
        print(f"  {i}. [{memory.node_type}] {memory.content[:50]}...")
    
    # Get version chain
    print(f"\n📜 Version chain for preference:")
    version_chain = engine.get_memory_chain(preference_id)
    print(f"Found {len(version_chain)} versions:")
    for i, version in enumerate(version_chain, 1):
        print(f"  v{version.version}: {version.content[:50]}...")
        print(f"         Created: {version.timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    # System statistics
    print(f"\n📊 Memory system statistics:")
    stats = engine.get_memory_stats()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Active memories: {stats['active_memories']}")
    print(f"  Average activation: {stats['average_activation']:.3f}")
    print(f"  Memory types: {stats['memory_types']}")
    print(f"  Total tags: {stats['total_tags']}")
    print(f"  Total links: {stats['total_links']}")
    
    # Apply decay simulation
    print(f"\n⏰ Applying memory decay...")
    decayed_count = engine.apply_global_decay()
    print(f"Applied decay to {decayed_count} memories")
    
    # Show updated statistics
    updated_stats = engine.get_memory_stats()
    print(f"Updated average activation: {updated_stats['average_activation']:.3f}")
    
    print(f"\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
