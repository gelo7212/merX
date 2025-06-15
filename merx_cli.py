#!/usr/bin/env python3
"""
merX Memory System - Command Line Interface

A production-ready CLI tool for interacting with the merX memory system.
Supports both testing and production environments with comprehensive
logging, configuration management, and error handling.
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime
import signal
import atexit

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.factory.enhanced_memory_factory import EnhancedMemoryEngineFactory


class MerXConfig:
    """Configuration management for merX CLI"""

    DEFAULT_CONFIG = {
        "data_path": "data/dev/hp_mini.mex",
        "ram_capacity": 100000,
        "flush_interval": 5.0,
        "flush_threshold": 100,
        "log_level": "INFO",
        "log_file": "logs/merx_cli.log",
        "enable_compression": True,
        "compression_level": 5,
        "enable_distributed": False,
        "shard_count": 4,
        "activation_threshold": 0.01,
        "spreading_decay": 0.7,
        "max_hops": 3,
        "decay_interval": 60,
        "performance_monitoring": True,
        "auto_backup": True,
        "backup_interval": 3600,
    }

    def __init__(
        self, config_file: Optional[str] = None, environment: str = "development"
    ):
        """
        Initialize configuration.

        Args:
            config_file: Path to configuration file
            environment: Environment type (development, testing, production)
        """
        self.environment = environment
        self.config = self.DEFAULT_CONFIG.copy()

        # Environment-specific defaults
        if environment == "production":
            self.config.update(
                {
                    "data_path": "data/prod/hp_mini.mex",
                    "ram_capacity": 500000,
                    "log_level": "WARNING",
                    "performance_monitoring": True,
                    "auto_backup": True,
                    "flush_interval": 2.0,
                    "flush_threshold": 50,
                }
            )
        elif environment == "testing":
            self.config.update(
                {
                    "ram_capacity": 500000,
                    "log_level": "DEBUG",
                    "data_path": "data/test_output/hp_mini.mex",
                    "flush_interval": 1.0,
                    "flush_threshold": 10,
                }
            )

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)
                self.config.update(file_config)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_file}: {e}")

    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save config to {config_file}: {e}")

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value


class MerXCLI:
    """Production-ready Command Line Interface for merX Memory System"""

    def __init__(self, config: MerXConfig):
        """
        Initialize the CLI with memory engine.

        Args:
            config: Configuration object
        """
        self.config = config
        self.engine = None
        self.logger = self._setup_logging()
        self.stats = {
            "operations": 0,
            "errors": 0,
            "start_time": time.time(),
            "last_operation": None,
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)

        # Initialize memory engine
        self._initialize_engine()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("merx_cli")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))

        # Create logs directory
        log_file = self.config.get("log_file", "logs/merx_cli.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.get("log_level", "INFO")))

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _initialize_engine(self):
        """Initialize the memory engine with configuration."""
        data_path = self.config.get("data_path")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        self.logger.info(
            f"Initializing merX Memory System for {self.config.environment}"
        )   
        
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"RAM capacity: {self.config.get('ram_capacity'):,} nodes")

        try:
            # Create engine with only supported parameters
            self.engine = EnhancedMemoryEngineFactory.create_engine(
                data_path=data_path,
                ram_capacity=self.config.get("ram_capacity"),
                flush_interval=self.config.get("flush_interval"),
                flush_threshold=self.config.get("flush_threshold")
            )

            self.logger.info("Memory engine initialized successfully")
            if self.config.environment != "production":
                print("✅ Memory engine initialized successfully!")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory engine: {e}")
            print(f"❌ Failed to initialize memory engine: {e}")
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)

    def insert_memory(
        self, content: str, tags: Optional[List[str]] = None, node_type: str = "fact"
    ) -> Optional[UUID]:
        """
        Insert a new memory into the system.

        Args:
            content: Memory content
            tags: Optional list of tags
            node_type: Type of memory (fact, procedure, episodic, task)

        Returns:
            UUID of inserted memory or None if failed
        """
        if not self.engine:
            self.logger.error("Engine not initialized")
            return None

        try:
            start_time = time.time()

            memory_id = self.engine.insert_memory(
                content=content, tags=tags or [], node_type=node_type
            )

            insert_time = (time.time() - start_time) * 1000
            self.stats["operations"] += 1
            self.stats["last_operation"] = "insert"

            self.logger.info(f"Memory inserted: {memory_id} ({insert_time:.2f}ms)")

            if self.config.environment != "production":
                print(f"✅ Memory inserted successfully!")
                print(f"   ID: {memory_id}")
                print(f"   Content: {content[:60]}{'...' if len(content) > 60 else ''}")
                print(f"   Tags: {tags or 'None'}")
                print(f"   Type: {node_type}")
                print(f"   Time: {insert_time:.2f}ms")

            return memory_id

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Failed to insert memory: {e}")
            if self.config.environment != "production":
                print(f"❌ Failed to insert memory: {e}")
            return None

    def search_memories(
        self, query: str = "", limit: int = 10, tags: Optional[List[str]] = None
    ):
        """
        Search for memories using content or tags.

        Args:
            query: Search query
            limit: Maximum number of results
            tags: Optional tags to filter by
        """
        if not self.engine:
            self.logger.error("Engine not initialized")
            return

        try:
            start_time = time.time()

            if tags:
                results = self.engine.recall_memories(tags=tags, limit=limit)
                search_type = f"tag search: {tags}"
            else:
                results = self.engine.recall_memories(query=query, limit=limit)
                search_type = f"content search: '{query}'"

            search_time = (time.time() - start_time) * 1000
            self.stats["operations"] += 1
            self.stats["last_operation"] = "search"

            self.logger.info(
                f"Search completed: {len(results)} results ({search_time:.2f}ms)"
            )

            if self.config.environment != "production":
                print(f"\n🔍 Search Results ({search_type})")
                print("=" * 60)
                print(f"Found {len(results)} results in {search_time:.2f}ms")

                if not results:
                    print("No memories found matching your query.")
                    return

                for i, memory in enumerate(results, 1):
                    print(f"\n{i}. Memory ID: {memory.id}")
                    print(f"   Content: {memory.content}")
                    print(f"   Tags: {memory.tags}")
                    print(f"   Type: {memory.node_type}")
                    print(f"   Activation: {memory.activation:.3f}")
                    print(f"   Created: {memory.timestamp}")

            return results

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Search failed: {e}")
            if self.config.environment != "production":
                print(f"❌ Search failed: {e}")
            return []

    def get_memory(self, memory_id: str):
        """Get a specific memory by ID."""
        if not self.engine:
            self.logger.error("Engine not initialized")
            return None

        try:
            memory = self.engine.get_memory(UUID(memory_id))

            if memory:
                self.logger.info(f"Retrieved memory: {memory_id}")

                if self.config.environment != "production":
                    print(f"\n📄 Memory Details")
                    print("=" * 40)
                    print(f"ID: {memory.id}")
                    print(f"Content: {memory.content}")
                    print(f"Tags: {memory.tags}")
                    print(f"Type: {memory.node_type}")
                    print(f"Version: {memory.version}")
                    print(f"Activation: {memory.activation:.3f}")
                    print(f"Decay Rate: {memory.decay_rate:.3f}")
                    print(f"Created: {memory.timestamp}")

                    if memory.links:
                        print(f"Links: {len(memory.links)} connections")
                        print("\nConnected to:")
                        for link in memory.links:
                            print(
                                f"  → {link.to_id} (weight: {link.weight:.3f}, type: {link.link_type})"
                            )
                    else:
                        print("Links: No connections")
            else:
                self.logger.warning(f"Memory not found: {memory_id}")
                if self.config.environment != "production":
                    print(f"❌ Memory with ID {memory_id} not found")

            return memory

        except ValueError:
            self.logger.error(f"Invalid UUID format: {memory_id}")
            if self.config.environment != "production":
                print(f"❌ Invalid UUID format: {memory_id}")
            return None
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Failed to get memory: {e}")
            if self.config.environment != "production":
                print(f"❌ Failed to get memory: {e}")
            return None

    def show_stats(self):
        """Display system statistics."""
        if not self.engine:
            self.logger.error("Engine not initialized")
            return

        try:
            stats = self.engine.get_memory_stats()
            runtime = time.time() - self.stats["start_time"]

            self.logger.info("Statistics requested")

            if self.config.environment != "production":
                print(f"\n📊 merX System Statistics")
                print("=" * 40)
                print(f"Environment: {self.config.environment}")
                print(f"Runtime: {runtime:.1f}s")
                print(f"Operations: {self.stats['operations']}")
                print(f"Errors: {self.stats['errors']}")
                print(
                    f"Success rate: {((self.stats['operations'] - self.stats['errors']) / max(1, self.stats['operations']) * 100):.1f}%"
                )

                print(f"\nMemory Statistics:")
                print(f"Total memories: {stats.get('total_memories', 'N/A'):,}")
                print(f"RAM usage: {stats.get('ram_usage_mb', 'N/A')} MB")
                print(f"Active nodes: {stats.get('active_nodes', 'N/A'):,}")
                print(f"Cache hit rate: {stats.get('cache_hit_rate', 'N/A')}")
                print(f"Avg query time: {stats.get('avg_query_time', 'N/A')} ms")

                # Additional RAMX stats if available
                try:
                    if hasattr(self.engine.storage, "ramx"):
                        ramx_stats = self.engine.storage.ramx.get_stats()
                        print(f"\n💾 RAMX Statistics")
                        print(f"RAMX nodes: {ramx_stats.get('total_nodes', 'N/A'):,}")
                        print(
                            f"Word index size: {ramx_stats.get('word_index_size', 'N/A'):,}"
                        )
                        print(
                            f"Tag index size: {ramx_stats.get('tag_index_size', 'N/A'):,}"
                        )
                        print(
                            f"Average activation: {ramx_stats.get('avg_activation', 'N/A'):.3f}"
                        )
                except Exception as ex:
                    self.logger.debug(f"Could not get RAMX stats: {ex}")

            return stats

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Failed to get statistics: {e}")
            if self.config.environment != "production":
                print(f"❌ Failed to get statistics: {e}")
            return None

    def run_health_check(self) -> bool:
        """Run a comprehensive health check."""
        if not self.engine:
            return False

        try:
            self.logger.info("Running health check...")

            # Test insert
            test_id = self.insert_memory(
                f"Health check test {int(time.time())}", ["health", "test"]
            )
            if not test_id:
                return False

            # Test retrieval
            memory = self.get_memory(str(test_id))
            if not memory:
                return False

            # Test search
            results = self.search_memories("health check", limit=5)
            if not results:
                return False

            # Check system resources
            stats = self.show_stats()
            if not stats:
                return False

            self.logger.info("Health check passed")
            if self.config.environment != "production":
                print("✅ Health check passed!")

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            if self.config.environment != "production":
                print(f"❌ Health check failed: {e}")
            return False

    def backup_data(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the memory data."""
        if not self.engine:
            return False

        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/memory_backup_{timestamp}.mex"

            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # Simple file copy for now
            import shutil

            data_path = self.config.get("data_path")

            if data_path and os.path.exists(data_path):
                shutil.copy2(data_path, backup_path)

                self.logger.info(f"Data backed up to: {backup_path}")
                if self.config.environment != "production":
                    print(f"✅ Data backed up to: {backup_path}")

                return True
            else:
                self.logger.warning(f"Source data file not found: {data_path}")
                if self.config.environment != "production":
                    print(f"⚠️ Source data file not found: {data_path}")
                return False

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            if self.config.environment != "production":
                print(f"❌ Backup failed: {e}")
            return False

    def interactive_mode(self):
        """Start interactive mode for continuous interaction."""
        if self.config.environment != "production":
            print(f"\n🎮 merX Interactive Mode ({self.config.environment})")
            print("=" * 50)
            print("Commands:")
            print(
                "  insert <content> [#tag1 #tag2]  - Insert memory with content and tags"
            )
            print("  search <query>                  - Search memories by content")
            print("  search --tags tag1,tag2         - Search by tags")
            print("  get <memory_id>                 - Get memory by ID")
            print("  stats                           - Show system statistics")
            print("  health                          - Run health check")
            print("  backup [path]                   - Create data backup")
            print("  config                          - Show configuration")
            print("  help                            - Show detailed help")
            print("  quit                            - Exit interactive mode")

        while True:
            try:
                if self.config.environment == "production":
                    user_input = input("merX> ").strip()
                else:
                    user_input = input(f"\nmerX ({self.config.environment})> ").strip()

                if not user_input:
                    continue

                self._process_command(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Command processing error: {e}")
                if self.config.environment != "production":
                    print(f"❌ Error: {e}")

    def _process_command(self, user_input: str):
        """Process a single command."""
        parts = user_input.split(" ", 1)
        command = parts[0].lower()

        if command in ["quit", "exit"]:
            raise KeyboardInterrupt()
        elif command == "help":
            self._show_interactive_help()
        elif command == "stats":
            self.show_stats()
        elif command == "health":
            self.run_health_check()
        elif command == "config":
            self._show_config()
        elif command == "backup":
            backup_path = parts[1] if len(parts) > 1 else None
            self.backup_data(backup_path)
        elif command == "insert" and len(parts) > 1:
            content = parts[1]
            tags = []
            node_type = "fact"

            # Parse hashtags from content
            if "#" in content:
                content_parts = content.split("#")
                content = content_parts[0].strip()
                for tag_part in content_parts[1:]:
                    tag_words = tag_part.split()
                    if tag_words:
                        tags.append(tag_words[0])

            self.insert_memory(content, tags, node_type)
        elif command == "search" and len(parts) > 1:
            query_part = parts[1]
            if query_part.startswith("--tags "):
                tags = query_part[7:].split(",")
                self.search_memories("", tags=tags)
            else:
                self.search_memories(query_part)
        elif command == "get" and len(parts) > 1:
            memory_id = parts[1]
            self.get_memory(memory_id)
        else:
            if self.config.environment != "production":
                print("❌ Unknown command. Type 'help' for available commands.")

    def _show_config(self):
        """Show current configuration."""
        if self.config.environment != "production":
            print(f"\n⚙️ Current Configuration")
            print("=" * 40)
            for key, value in self.config.config.items():
                print(f"{key}: {value}")

    def _show_interactive_help(self):
        """Show detailed help for interactive mode."""
        if self.config.environment != "production":
            print(f"\n📖 merX Interactive Mode - Detailed Help")
            print("=" * 60)
            print("INSERT MEMORY:")
            print("  insert Hello world #greeting #test")
            print("  insert Python is great for AI #programming #python #AI")
            print("  insert Remember to buy milk #todo #shopping")
            print("\nSEARCH MEMORIES:")
            print("  search programming")
            print("  search AI python")
            print("  search --tags todo,shopping")
            print("\nGET SPECIFIC MEMORY:")
            print("  get 12345678-1234-5678-9012-123456789abc")
            print("\nSYSTEM COMMANDS:")
            print("  stats  - Show system statistics")
            print("  health - Run health check")
            print("  backup - Create data backup")
            print("  config - Show configuration")
            print("  help   - Show this help")
            print("  quit   - Exit interactive mode")
            print("\nTips:")
            print("• Use #hashtags in content to automatically add tags")
            print("• Search is fuzzy and works with partial matches")
            print("• All memories are automatically saved to disk")
            print("• Use health command to verify system integrity")

    def shutdown(self):
        """Properly shutdown the memory system."""
        if self.engine:
            try:
                self.logger.info("Shutting down memory system...")

                if self.config.environment != "production":
                    print("\n🔄 Shutting down memory system...")

                EnhancedMemoryEngineFactory.cleanup_and_shutdown(self.engine)

                runtime = time.time() - self.stats["start_time"]
                self.logger.info(
                    f"Shutdown complete. Runtime: {runtime:.1f}s, Operations: {self.stats['operations']}, Errors: {self.stats['errors']}"
                )

                if self.config.environment != "production":
                    print("✅ Shutdown complete!")

            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
                if self.config.environment != "production":
                    print(f"❌ Error during shutdown: {e}")
            finally:
                self.engine = None


def main():
    """Main CLI function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="merX Memory System - Production-Ready Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Types:
  development  - Default settings with verbose output
  testing      - Small RAM capacity, debug logging, test data path
  production   - High capacity, minimal output, warning-level logging

Examples:
  python merx_cli.py --interactive                           # Development mode
  python merx_cli.py --env production --interactive          # Production mode
  python merx_cli.py --env testing --run-tests               # Testing mode
  python merx_cli.py --insert "Python is great" --tags python,programming
  python merx_cli.py --search "programming language"
  python merx_cli.py --config config.json --interactive
  python merx_cli.py --data-path "custom/path.mex" --env production --stats

Interactive Mode Commands:
  insert Hello world #greeting #test
  search programming
  search --tags todo,shopping
  get 12345678-1234-5678-9012-123456789abc
  health
  backup
  stats
  config
  help
  quit
""",
    )

    # Environment and configuration
    parser.add_argument(
        "--env",
        "--environment",
        choices=["development", "testing", "production"],
        default="development",
        help="Environment type (default: development)",
    )

    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")

    parser.add_argument("--data-path", type=str, help="Override data path from config")

    parser.add_argument(
        "--ram-capacity", type=int, help="Override RAM capacity from config"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )

    # Action options
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )

    parser.add_argument("--insert", type=str, help="Insert memory with given content")

    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags for insertion (use with --insert)",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="fact",
        choices=["fact", "procedure", "episodic", "task"],
        help="Memory type (default: fact)",
    )

    parser.add_argument(
        "--search", type=str, help="Search for memories with given query"
    )

    parser.add_argument(
        "--search-tags", type=str, help="Search by comma-separated tags"
    )

    parser.add_argument("--get", type=str, help="Get memory by UUID")

    parser.add_argument("--stats", action="store_true", help="Show system statistics")

    parser.add_argument("--health", action="store_true", help="Run system health check")

    parser.add_argument(
        "--backup",
        type=str,
        nargs="?",
        const="auto",
        help="Create data backup (optional: specify path)",
    )

    parser.add_argument(
        "--run-tests", action="store_true", help="Run comprehensive system tests"
    )

    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum search results (default: 10)"
    )

    parser.add_argument(
        "--save-config", type=str, help="Save current configuration to file"
    )

    args = parser.parse_args()

    # Show help if no arguments
    if len(sys.argv) == 1:
        print("🧠 merX Memory System - Production-Ready CLI")
        print("=" * 60)
        print("Quick start:")
        print("  Development: python merx_cli.py --interactive")
        print("  Production:  python merx_cli.py --env production --interactive")
        print("  Testing:     python merx_cli.py --env testing --run-tests")
        print("")
        parser.print_help()
        return

    # Initialize configuration
    try:
        config = MerXConfig(config_file=args.config, environment=args.env)

        # Apply command line overrides
        if args.data_path:
            config.set("data_path", args.data_path)
        if args.ram_capacity:
            config.set("ram_capacity", args.ram_capacity)
        if args.log_level:
            config.set("log_level", args.log_level)

        # Save config if requested
        if args.save_config:
            config.save_to_file(args.save_config)
            print(f"✅ Configuration saved to {args.save_config}")
            return

        # Initialize CLI
        cli = MerXCLI(config)

        # Execute requested actions
        action_taken = False

        if args.insert:
            tags = args.tags.split(",") if args.tags else None
            result = cli.insert_memory(args.insert, tags, args.type)
            action_taken = True
            if not result and config.environment == "production":
                sys.exit(1)

        if args.search:
            results = cli.search_memories(args.search, args.limit)
            action_taken = True
            if config.environment == "production" and results:
                # Output JSON for production use
                output = [
                    {"id": str(r.id), "content": r.content, "activation": r.activation}
                    for r in results
                ]
                print(json.dumps(output, indent=2))

        if args.search_tags:
            tags = args.search_tags.split(",")
            results = cli.search_memories("", args.limit, tags=tags)
            action_taken = True
            if config.environment == "production" and results:
                output = [
                    {"id": str(r.id), "content": r.content, "tags": r.tags}
                    for r in results
                ]
                print(json.dumps(output, indent=2))

        if args.get:
            memory = cli.get_memory(args.get)
            action_taken = True
            if memory and config.environment == "production":
                output = {
                    "id": str(memory.id),
                    "content": memory.content,
                    "tags": memory.tags,
                    "type": memory.node_type,
                    "activation": memory.activation,
                    "timestamp": str(memory.timestamp),
                }
                print(json.dumps(output, indent=2))

        if args.stats:
            stats = cli.show_stats()
            action_taken = True
            if stats and config.environment == "production":
                print(json.dumps(stats, indent=2, default=str))

        if args.health:
            success = cli.run_health_check()
            action_taken = True
            if config.environment == "production":
                print(json.dumps({"health_check": "passed" if success else "failed"}))
                if not success:
                    sys.exit(1)

        if args.backup:
            backup_path = None if args.backup == "auto" else args.backup
            success = cli.backup_data(backup_path)
            action_taken = True
            if not success and config.environment == "production":
                sys.exit(1)

        if args.run_tests:
            success = run_comprehensive_tests(cli)
            action_taken = True
            if not success:
                sys.exit(1)

        if args.interactive:
            cli.interactive_mode()
            action_taken = True

        # If no specific action was taken, show brief status
        if not action_taken:
            if config.environment != "production":
                print("merX CLI initialized. Use --help for available commands.")
            cli.show_stats()

        # Shutdown
        cli.shutdown()

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        if args.env == "production":
            print(json.dumps({"error": str(e)}))
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)


def run_comprehensive_tests(cli: MerXCLI) -> bool:
    """Run comprehensive system tests."""
    print("🧪 Running comprehensive merX tests...")

    tests_passed = 0
    total_tests = 6

    try:
        # Test 1: Basic insertion
        print("\n1. Testing memory insertion...")
        test_id = cli.insert_memory(
            "Test memory for automated testing", ["test", "automated"], "fact"
        )
        if test_id:
            print("✅ Memory insertion test passed")
            tests_passed += 1
        else:
            print("❌ Memory insertion test failed")

        # Test 2: Memory retrieval
        print("\n2. Testing memory retrieval...")
        if test_id:
            memory = cli.get_memory(str(test_id))
            if memory and memory.content == "Test memory for automated testing":
                print("✅ Memory retrieval test passed")
                tests_passed += 1
            else:
                print("❌ Memory retrieval test failed")

        # Test 3: Content search
        print("\n3. Testing content search...")
        results = cli.search_memories("automated testing", limit=5)
        if results and len(results) > 0:
            print("✅ Content search test passed")
            tests_passed += 1
        else:
            print("❌ Content search test failed")

        # Test 4: Tag search
        print("\n4. Testing tag search...")
        results = cli.search_memories("", tags=["test"])
        if results and len(results) > 0:
            print("✅ Tag search test passed")
            tests_passed += 1
        else:
            print("❌ Tag search test failed")

        # Test 5: Statistics
        print("\n5. Testing statistics...")
        stats = cli.show_stats()
        if stats:
            print("✅ Statistics test passed")
            tests_passed += 1
        else:
            print("❌ Statistics test failed")

        # Test 6: Health check
        print("\n6. Testing health check...")
        health = cli.run_health_check()
        if health:
            print("✅ Health check test passed")
            tests_passed += 1
        else:
            print("❌ Health check test failed")

        # Results
        success_rate = (tests_passed / total_tests) * 100
        print(
            f"\n📊 Test Results: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)"
        )

        if tests_passed == total_tests:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️ Some tests failed!")
            return False

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return False


if __name__ == "__main__":
    main()
