"""
Utility functions for the merX memory system.
"""

import uuid
import time
import logging
import hashlib
from typing import List, Set
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID for memory nodes."""
    return uuid.uuid4()


def current_timestamp() -> datetime:
    """Get current timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


def timestamp_to_float(dt: datetime) -> float:
    """Convert datetime to Unix timestamp float."""
    return dt.timestamp()


def float_to_timestamp(timestamp: float) -> datetime:
    """Convert Unix timestamp float to datetime."""
    return datetime.fromtimestamp(timestamp, timezone.utc)


def content_hash(content: str) -> str:
    """Generate a hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def extract_simple_tags(content: str, max_tags: int = 10) -> List[str]:
    """
    Extract simple tags from content.
    
    This is a basic implementation that could be enhanced with NLP libraries.
    """
    import re
    
    # Convert to lowercase for processing
    text = content.lower()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-z]{3,}\b', text)
    
    # Filter out stop words and duplicates
    meaningful_words = []
    seen = set()
    
    for word in words:
        if word not in stop_words and word not in seen and len(word) >= 3:
            meaningful_words.append(word)
            seen.add(word)
            
            if len(meaningful_words) >= max_tags:
                break
    
    return meaningful_words


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple similarity score between two texts.
    
    Uses Jaccard similarity on word sets.
    """
    if not text1 or not text2:
        return 0.0
    
    # Extract word sets
    words1 = set(extract_simple_tags(text1, max_tags=50))
    words2 = set(extract_simple_tags(text2, max_tags=50))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def validate_weight(weight: float) -> float:
    """Validate and normalize a weight value."""
    if not isinstance(weight, (int, float)):
        raise ValueError(f"Weight must be numeric, got {type(weight)}")
    
    weight = float(weight)
    
    if weight < 0.0:
        logger.warning(f"Negative weight {weight} clamped to 0.0")
        return 0.0
    elif weight > 1.0:
        logger.warning(f"Weight {weight} > 1.0 clamped to 1.0")
        return 1.0
    
    return weight


def validate_activation(activation: float) -> float:
    """Validate and normalize an activation value."""
    return validate_weight(activation)  # Same validation rules


def format_memory_summary(content: str, max_length: int = 100) -> str:
    """Format memory content for display in summaries."""
    if len(content) <= max_length:
        return content
    
    # Try to break at word boundary
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # If we can break reasonably close to the end
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def memory_age_in_hours(timestamp: datetime) -> float:
    """Calculate the age of a memory in hours."""
    now = current_timestamp()
    age = now - timestamp
    return age.total_seconds() / 3600


def benchmark_operation(operation_name: str):
    """
    Decorator to benchmark memory operations.
    
    Usage:
        @benchmark_operation("insert_memory")
        def some_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.debug(f"{operation_name} took {duration:.4f} seconds")
        return wrapper
    return decorator


class MemoryProfiler:
    """Simple profiler for memory operations."""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
    
    def record_operation(self, operation: str, duration: float):
        """Record the duration of an operation."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        
        for operation in self.operation_times:
            times = self.operation_times[operation]
            stats[operation] = {
                "count": self.operation_counts[operation],
                "total_time": sum(times),
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return stats
    
    def reset(self):
        """Reset all profiling data."""
        self.operation_times.clear()
        self.operation_counts.clear()


# Global profiler instance
profiler = MemoryProfiler()


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for use as a filename.
    
    Removes special characters and limits length.
    """
    import re
    
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    
    return sanitized or "unnamed"


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
