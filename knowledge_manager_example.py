#!/usr/bin/env python3
"""
merX Integration Example: Personal Knowledge Manager

This example demonstrates how to integrate merX into a personal knowledge
management application using the external API.

Features demonstrated:
1. Document storage and tagging
2. Intelligent search and retrieval
3. Relationship discovery
4. Knowledge base statistics
5. Export functionality
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import the merX external API
sys.path.append(str(Path(__file__).parent))
from merx_api import MerXMemory, create_memory_system

class PersonalKnowledgeManager:
    """
    A personal knowledge management system built on merX.
    
    This demonstrates how to create a higher-level application
    using the merX external API.
    """
    
    def __init__(self, knowledge_base_path: str = "data/personal_kb.mex"):
        """Initialize the knowledge manager."""
        self.kb_path = knowledge_base_path
        self.memory = MerXMemory(
            data_path=knowledge_base_path,
            ram_capacity=25000,  # Moderate cache for personal use
            auto_flush=True,
            log_level="INFO"
        )
        
        print(f"Personal Knowledge Base initialized: {knowledge_base_path}")
    
    def add_note(self, content: str, category: str = "general", 
                 tags: List[str] = None, related_notes: List[str] = None) -> str:
        """
        Add a new note to the knowledge base.
        
        Args:
            content: The note content
            category: Note category (research, personal, work, etc.)
            tags: Additional tags for organization
            related_notes: IDs of related existing notes
        
        Returns:
            Note ID
        """
        # Prepare tags
        all_tags = [category]
        if tags:
            all_tags.extend(tags)
        
        # Store the note
        note_id = self.memory.store(
            content=content,
            memory_type="note",
            tags=all_tags,
            related_to=related_notes
        )
        
        print(f"Added note: {note_id[:8]}... (category: {category})")
        return note_id
    
    def add_web_article(self, title: str, url: str, summary: str, 
                       tags: List[str] = None) -> str:
        """Add a web article reference to the knowledge base."""
        content = f"Title: {title}\nURL: {url}\nSummary: {summary}"
        
        article_tags = ["web-article", "reference"]
        if tags:
            article_tags.extend(tags)
        
        return self.memory.store(
            content=content,
            memory_type="reference",
            tags=article_tags
        )
    
    def add_book_note(self, book_title: str, author: str, page: int, 
                     note_content: str, tags: List[str] = None) -> str:
        """Add a note from a book."""
        content = f"Book: {book_title} by {author} (p. {page})\nNote: {note_content}"
        
        book_tags = ["book", "reading", author.lower().replace(" ", "-")]
        if tags:
            book_tags.extend(tags)
        
        return self.memory.store(
            content=content,
            memory_type="book-note",
            tags=book_tags
        )
    
    def search_knowledge(self, query: str, category: str = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            category: Limit to specific category
            limit: Maximum results
        
        Returns:
            List of matching notes
        """
        tags = [category] if category else None
        
        results = self.memory.search(
            query=query,
            tags=tags,
            limit=limit,
            search_mode="balanced"
        )
        
        print(f"Found {len(results)} results for '{query}'")
        return results
    
    def explore_topic(self, topic_query: str, depth: int = 2) -> Dict[str, Any]:
        """
        Explore a topic by finding related information.
        
        Args:
            topic_query: Topic to explore
            depth: How deep to search for relationships
        
        Returns:
            Dictionary with exploration results
        """
        # First, find initial matches
        initial_results = self.search_knowledge(topic_query, limit=5)
        
        if not initial_results:
            return {"query": topic_query, "initial_results": [], "related": []}
        
        # Find related notes for the most relevant result
        primary_note = initial_results[0]
        related_notes = self.memory.find_related(primary_note["id"], max_depth=depth)
        
        return {
            "query": topic_query,
            "initial_results": initial_results,
            "related": related_notes,
            "total_found": len(initial_results) + len(related_notes)
        }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        stats = self.memory.get_stats()
        
        # Add custom analytics
        all_data = self.memory.export_data()
        memories = all_data.get("memories", [])
        
        # Category breakdown
        categories = {}
        note_types = {}
        tag_frequency = {}
        
        for memory in memories:
            # Count by first tag (usually category)
            if memory["tags"]:
                category = memory["tags"][0]
                categories[category] = categories.get(category, 0) + 1
            
            # Count by type
            mem_type = memory["type"]
            note_types[mem_type] = note_types.get(mem_type, 0) + 1
            
            # Tag frequency
            for tag in memory["tags"]:
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
        
        # Sort tags by frequency
        top_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "system_stats": stats,
            "total_notes": len(memories),
            "categories": categories,
            "note_types": note_types,
            "top_tags": top_tags,
            "data_path": self.kb_path
        }
    
    def export_knowledge_base(self, format: str = "json") -> Dict[str, Any]:
        """Export the entire knowledge base."""
        return self.memory.export_data(format=format)
    
    def cleanup(self):
        """Clean up the knowledge manager."""
        self.memory.cleanup()

def demo_personal_knowledge_manager():
    """Demonstration of the Personal Knowledge Manager."""
    print("=== merX Personal Knowledge Manager Demo ===\n")
    
    # Initialize knowledge manager
    kb = PersonalKnowledgeManager("data/demo_personal_kb.mex")
    
    try:
        print("1. Adding various types of notes...")
        
        # Add research notes
        research_id = kb.add_note(
            "Machine learning models require careful feature engineering for optimal performance",
            category="research",
            tags=["AI", "machine-learning", "features"]
        )
        
        # Add a web article
        article_id = kb.add_web_article(
            title="The Future of Artificial Intelligence",
            url="https://example.com/ai-future",
            summary="Discusses upcoming trends in AI including neural networks and automation",
            tags=["AI", "trends", "automation"]
        )
        
        # Add book notes
        book_id = kb.add_book_note(
            book_title="Pattern Recognition and Machine Learning",
            author="Christopher Bishop",
            page=45,
            note_content="Bayesian inference provides a principled approach to uncertainty quantification",
            tags=["statistics", "bayesian", "uncertainty"]
        )
        
        # Add personal insights
        insight_id = kb.add_note(
            "The key to successful AI projects is understanding the business problem first, then applying appropriate technical solutions",
            category="insights",
            tags=["AI", "business", "strategy"],
            related_notes=[research_id]
        )
        
        print(f"\nAdded 4 notes to knowledge base")
        
        print("\n2. Searching the knowledge base...")
        
        # Search for AI-related content
        ai_results = kb.search_knowledge("artificial intelligence", limit=5)
        print(f"\nAI-related notes:")
        for i, result in enumerate(ai_results, 1):
            print(f"  {i}. {result['content'][:80]}...")
            print(f"     Tags: {result['tags']}")
        
        print("\n3. Exploring a topic in depth...")
        
        # Explore machine learning topic
        ml_exploration = kb.explore_topic("machine learning", depth=2)
        print(f"\nMachine Learning exploration:")
        print(f"  Initial results: {len(ml_exploration['initial_results'])}")
        print(f"  Related notes: {len(ml_exploration['related'])}")
        print(f"  Total found: {ml_exploration['total_found']}")
        
        # Show related notes
        if ml_exploration['related']:
            print("\n  Related notes:")
            for i, related in enumerate(ml_exploration['related'][:3], 1):
                print(f"    {i}. {related['content'][:60]}... (score: {related['activation_score']:.3f})")
        
        print("\n4. Knowledge base statistics...")
        
        stats = kb.get_knowledge_stats()
        print(f"\nKnowledge Base Statistics:")
        print(f"  Total notes: {stats['total_notes']}")
        print(f"  Categories: {list(stats['categories'].keys())}")
        print(f"  Note types: {list(stats['note_types'].keys())}")
        print(f"  Top tags: {[tag for tag, count in stats['top_tags'][:5]]}")
        
        print("\n5. Exporting knowledge base...")
        
        export_data = kb.export_knowledge_base()
        print(f"  Exported {len(export_data['memories'])} memories")
        print(f"  Export timestamp: {export_data['metadata']['export_time']}")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        kb.cleanup()

def interactive_knowledge_manager():
    """Interactive command-line interface for the knowledge manager."""
    kb = PersonalKnowledgeManager()
    
    print("=== Interactive Personal Knowledge Manager ===")
    print("Commands: add, search, explore, stats, export, quit")
    print("Type 'help' for detailed command information\n")
    
    try:
        while True:
            command = input("kb> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            elif command == "help":
                print("""
Commands:
  add     - Add a new note
  search  - Search the knowledge base
  explore - Explore a topic with related notes
  stats   - Show knowledge base statistics
  export  - Export knowledge base to JSON
  quit    - Exit the application
                """)
            elif command == "add":
                content = input("Note content: ")
                category = input("Category (default: general): ") or "general"
                tags_input = input("Tags (comma-separated): ")
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                
                note_id = kb.add_note(content, category, tags)
                print(f"Added note: {note_id[:8]}...")
                
            elif command == "search":
                query = input("Search query: ")
                category = input("Category filter (optional): ") or None
                
                results = kb.search_knowledge(query, category)
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['content'][:100]}...")
                    print(f"   ID: {result['id'][:8]}... | Tags: {result['tags']}")
                    
            elif command == "explore":
                topic = input("Topic to explore: ")
                exploration = kb.explore_topic(topic)
                
                print(f"\nExploration results for '{topic}':")
                print(f"Initial matches: {len(exploration['initial_results'])}")
                print(f"Related notes: {len(exploration['related'])}")
                
                for i, result in enumerate(exploration['initial_results'], 1):
                    print(f"\n{i}. {result['content'][:80]}...")
                    
            elif command == "stats":
                stats = kb.get_knowledge_stats()
                print(f"\nKnowledge Base Statistics:")
                print(f"Total notes: {stats['total_notes']}")
                print(f"Categories: {stats['categories']}")
                print(f"Top tags: {dict(stats['top_tags'][:5])}")
                
            elif command == "export":
                export_data = kb.export_knowledge_base()
                filename = f"kb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                import json
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                print(f"Exported to {filename}")
                
            else:
                print("Unknown command. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        kb.cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="merX Personal Knowledge Manager")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_personal_knowledge_manager()
    elif args.interactive:
        interactive_knowledge_manager()
    else:
        print("merX Personal Knowledge Manager")
        print("Use --demo for demonstration or --interactive for interactive mode")
        print("Example: python knowledge_manager_example.py --demo")
