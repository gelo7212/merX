"""
Memory decay processing - simulates neural memory decay over time.
"""

import math
from datetime import datetime, timedelta
import logging

from src.interfaces import IDecayProcessor, MemoryNode

logger = logging.getLogger(__name__)


class DecayProcessor(IDecayProcessor):
    """
    Processes memory decay using various decay models.
    
    Supports multiple decay functions:
    - Linear decay: activation -= decay_rate * time_elapsed
    - Exponential decay: activation *= exp(-decay_rate * time_elapsed)
    - Power law decay: activation *= (1 + time_elapsed)^(-decay_rate)
    """
    
    def __init__(self, decay_model: str = "exponential", min_activation: float = 0.01):
        """
        Initialize decay processor.
        
        Args:
            decay_model: Type of decay ("linear", "exponential", "power_law")
            min_activation: Minimum activation level (memories below this are considered "forgotten")
        """
        self.decay_model = decay_model
        self.min_activation = min_activation
        
        # Validate decay model
        if decay_model not in ["linear", "exponential", "power_law"]:
            raise ValueError(f"Unknown decay model: {decay_model}")
        
        logger.info(f"Initialized decay processor with {decay_model} model")
    
    def apply_decay(self, node: MemoryNode, current_time: datetime) -> MemoryNode:
        """Apply decay to a memory node based on time elapsed."""
        time_elapsed = (current_time - node.timestamp).total_seconds()
        
        # Don't decay if no time has passed
        if time_elapsed <= 0:
            return node
        
        # Calculate new activation
        old_activation = node.activation
        new_activation = self._calculate_decayed_activation(
            node.activation, 
            node.decay_rate, 
            time_elapsed
        )
        
        # Ensure activation doesn't go below minimum
        new_activation = max(new_activation, self.min_activation)
        
        # Create updated node (immutable)
        updated_node = MemoryNode(
            id=node.id,
            content=node.content,
            node_type=node.node_type,
            version=node.version,
            timestamp=node.timestamp,
            activation=new_activation,
            decay_rate=node.decay_rate,
            version_of=node.version_of,
            links=node.links.copy(),
            tags=node.tags.copy()
        )
        
        if abs(old_activation - new_activation) > 0.001:
            logger.debug(f"Applied decay to {node.id}: {old_activation:.3f} -> {new_activation:.3f}")
        
        return updated_node
    
    def calculate_decay(self, node: MemoryNode, current_time: datetime) -> float:
        """Calculate the decay amount without applying it."""
        time_elapsed = (current_time - node.timestamp).total_seconds()
        
        if time_elapsed <= 0:
            return 0.0
        
        new_activation = self._calculate_decayed_activation(
            node.activation,
            node.decay_rate,
            time_elapsed
        )
        
        return node.activation - new_activation
    
    def refresh_activation(self, node: MemoryNode, boost: float = 0.1) -> MemoryNode:
        """Refresh/boost activation when memory is accessed."""
        # Calculate boost amount - diminishing returns for already high activation
        current_activation = node.activation
        available_boost = (1.0 - current_activation) * boost
        new_activation = min(1.0, current_activation + available_boost)
        
        # Create updated node with refreshed timestamp and activation
        updated_node = MemoryNode(
            id=node.id,
            content=node.content,
            node_type=node.node_type,
            version=node.version,
            timestamp=datetime.now(),  # Update timestamp to "now"
            activation=new_activation,
            decay_rate=node.decay_rate,
            version_of=node.version_of,
            links=node.links.copy(),
            tags=node.tags.copy()
        )
        
        logger.debug(f"Refreshed activation for {node.id}: {current_activation:.3f} -> {new_activation:.3f}")
        
        return updated_node
    
    def _calculate_decayed_activation(self, activation: float, decay_rate: float, time_elapsed: float) -> float:
        """Calculate decayed activation using the selected decay model."""
        
        if self.decay_model == "linear":
            # Linear decay: activation decreases linearly with time
            return max(0.0, activation - (decay_rate * time_elapsed))
        
        elif self.decay_model == "exponential":
            # Exponential decay: activation decays exponentially
            # Common in neuroscience and psychology
            return activation * math.exp(-decay_rate * time_elapsed)
        
        elif self.decay_model == "power_law":
            # Power law decay: activation follows power law
            # Often observed in human memory
            time_hours = time_elapsed / 3600  # Convert to hours
            return activation * math.pow(1 + time_hours, -decay_rate)
        
        else:
            raise ValueError(f"Unknown decay model: {self.decay_model}")
    
    def is_forgotten(self, node: MemoryNode) -> bool:
        """Check if a memory node is considered forgotten (below minimum activation)."""
        return node.activation < self.min_activation
    
    def estimate_forget_time(self, node: MemoryNode) -> timedelta:
        """Estimate when this memory will be forgotten."""
        if node.activation <= self.min_activation:
            return timedelta(0)
        
        if self.decay_model == "linear":
            # Time = (current_activation - min_activation) / decay_rate
            seconds = (node.activation - self.min_activation) / node.decay_rate
            return timedelta(seconds=seconds)
        
        elif self.decay_model == "exponential":
            # Time = -ln(min_activation/current_activation) / decay_rate
            if node.activation <= 0:
                return timedelta(0)
            seconds = -math.log(self.min_activation / node.activation) / node.decay_rate
            return timedelta(seconds=seconds)
        
        elif self.decay_model == "power_law":
            # Time = ((current_activation/min_activation)^(1/decay_rate) - 1) * 3600
            if node.activation <= 0:
                return timedelta(0)
            hours = math.pow(node.activation / self.min_activation, 1.0 / node.decay_rate) - 1
            return timedelta(hours=hours)
        
        else:
            return timedelta(days=365)  # Unknown model, return a year
