"""
Memory recall engine with spreading activation algorithm.
"""

import re
from typing import List, Dict, Set
from uuid import UUID
from collections import defaultdict
import logging

from src.interfaces import IRecallEngine, IMemoryStorage, IMemoryLinker, MemoryNode

logger = logging.getLogger(__name__)


class RecallEngine(IRecallEngine):
    """
    Implements memory recall using spreading activation and content similarity.

    Combines multiple recall strategies:
    - Content-based similarity matching
    - Tag-based filtering
    - Spreading activation through neural-like networks
    - Activation-based ranking
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        linker: IMemoryLinker,
        activation_threshold: float = 0.1,
        spreading_decay: float = 0.7,
    ):
        """
        Initialize recall engine.

        Args:
            storage: Memory storage interface
            linker: Memory linker for graph operations
            activation_threshold: Minimum activation for recall
            spreading_decay: Decay factor for spreading activation (0.0-1.0)
        """
        self.storage = storage
        self.linker = linker
        self.activation_threshold = activation_threshold
        self.spreading_decay = spreading_decay

        logger.info(
            f"Initialized recall engine (threshold={activation_threshold}, decay={spreading_decay})"
        )

    def recall_by_content(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """Recall memories by content similarity."""
        if not query.strip():
            return []

        all_nodes = self.storage.get_all_nodes()

        # Filter by activation threshold
        active_nodes = [
            node for node in all_nodes if node.activation >= self.activation_threshold
        ]

        # Calculate content similarity scores
        scored_nodes = []
        query_words = self._extract_words(query.lower())

        for node in active_nodes:
            similarity = self._calculate_content_similarity(query_words, node)
            if similarity > 0:
                scored_nodes.append((node, similarity))

        # Sort by combined score (similarity * activation)
        scored_nodes.sort(key=lambda x: x[1] * x[0].activation, reverse=True)

        # Return top results
        result = [node for node, _ in scored_nodes[:limit]]

        logger.debug(f"Content recall for '{query}': found {len(result)} matches")
        return result

    def recall_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryNode]:
        """Recall memories by tag matching."""
        if not tags:
            return []

        all_nodes = self.storage.get_all_nodes()

        # Filter by activation and tag matching
        matching_nodes = []
        tags_lower = [tag.lower() for tag in tags]

        for node in all_nodes:
            if node.activation < self.activation_threshold:
                continue

            # Calculate tag overlap
            node_tags_lower = [tag.lower() for tag in node.tags]
            overlap = len(set(tags_lower) & set(node_tags_lower))

            if overlap > 0:
                # Score based on tag overlap and activation
                score = (overlap / len(tags_lower)) * node.activation
                matching_nodes.append((node, score))

        # Sort by score
        matching_nodes.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        result = [node for node, _ in matching_nodes[:limit]]

        logger.debug(f"Tag recall for {tags}: found {len(result)} matches")
        return result

    def spreading_activation(
        self, start_nodes: List[UUID], max_depth: int = 3
    ) -> Dict[UUID, float]:
        """
        Perform spreading activation from starting nodes.

        Returns a dictionary mapping node IDs to their final activation levels.
        """
        # Initialize activation levels
        activation_levels: Dict[UUID, float] = defaultdict(float)

        # Set initial activation for start nodes
        for node_id in start_nodes:
            node = self.storage.read_node_by_id(node_id)
            if node:
                activation_levels[node_id] = node.activation

        # Spreading activation through multiple iterations
        for depth in range(max_depth):
            new_activations: Dict[UUID, float] = defaultdict(float)

            # For each currently activated node
            for node_id, current_activation in activation_levels.items():
                if current_activation < self.activation_threshold:
                    continue

                # Get outgoing links
                links = self.linker.get_linked_nodes(node_id, min_weight=0.1)

                # Spread activation to linked nodes
                for link in links:
                    # Calculate activation to spread
                    spread_amount = (
                        current_activation * link.weight * self.spreading_decay
                    )

                    # Add to target node's activation
                    new_activations[link.to_id] = max(
                        new_activations[link.to_id], spread_amount
                    )

            # Update activation levels (combine with existing)
            for node_id, new_activation in new_activations.items():
                activation_levels[node_id] = max(
                    activation_levels[node_id], new_activation
                )

            # Stop if no significant activation is spreading
            total_new_activation = sum(new_activations.values())
            if total_new_activation < 0.01:
                break

        # Filter out nodes below threshold
        filtered_activations = {
            node_id: activation
            for node_id, activation in activation_levels.items()
            if activation >= self.activation_threshold
        }

        logger.debug(
            f"Spreading activation from {len(start_nodes)} nodes: activated {len(filtered_activations)} total nodes"
        )
        return filtered_activations

    def get_top_activated(self, limit: int = 10) -> List[MemoryNode]:
        """Get the most activated memory nodes."""
        all_nodes = self.storage.get_all_nodes()

        # Filter by activation threshold and sort
        active_nodes = [
            node for node in all_nodes if node.activation >= self.activation_threshold
        ]

        active_nodes.sort(key=lambda x: x.activation, reverse=True)

        result = active_nodes[:limit]
        logger.debug(f"Top activated memories: returning {len(result)} nodes")
        return result

    def contextual_recall(
        self,
        query: str = None,
        tags: List[str] = None,
        context_nodes: List[UUID] = None,
        limit: int = 10,
    ) -> List[MemoryNode]:
        """
        Advanced recall combining multiple strategies.

        Args:
            query: Text query for content matching
            tags: Tags to match
            context_nodes: Nodes to use as context for spreading activation
            limit: Maximum number of results
        """
        candidate_nodes: Dict[UUID, float] = {}

        # Content-based recall
        if query:
            content_matches = self.recall_by_content(query, limit * 2)
            for node in content_matches:
                query_words = self._extract_words(query.lower())
                similarity = self._calculate_content_similarity(query_words, node)
                candidate_nodes[node.id] = (
                    candidate_nodes.get(node.id, 0) + similarity * 0.6
                )

        # Tag-based recall
        if tags:
            tag_matches = self.recall_by_tags(tags, limit * 2)
            for node in tag_matches:
                node_tags_lower = [tag.lower() for tag in node.tags]
                tags_lower = [tag.lower() for tag in tags]
                overlap_score = len(set(tags_lower) & set(node_tags_lower)) / len(
                    tags_lower
                )
                candidate_nodes[node.id] = (
                    candidate_nodes.get(node.id, 0) + overlap_score * 0.4
                )

        # Spreading activation from context
        if context_nodes:
            activation_levels = self.spreading_activation(context_nodes, max_depth=2)
            for node_id, activation in activation_levels.items():
                candidate_nodes[node_id] = (
                    candidate_nodes.get(node_id, 0) + activation * 0.3
                )

        # If no specific criteria, return top activated
        if not candidate_nodes:
            return self.get_top_activated(limit)

        # Combine with base activation levels
        final_scores = []
        for node_id, score in candidate_nodes.items():
            node = self.storage.read_node_by_id(node_id)
            if node and node.activation >= self.activation_threshold:
                final_score = score * node.activation
                final_scores.append((node, final_score))

        # Sort and return top results
        final_scores.sort(key=lambda x: x[1], reverse=True)
        result = [node for node, _ in final_scores[:limit]]

        logger.debug(f"Contextual recall: found {len(result)} results")
        return result

    def _extract_words(self, text: str) -> Set[str]:
        """Extract words from text for similarity matching."""
        # Simple word extraction - could be enhanced with stemming, etc.
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)

    def _calculate_content_similarity(
        self, query_words: Set[str], node: MemoryNode
    ) -> float:
        """Calculate content similarity between query and node."""
        node_words = self._extract_words(node.content.lower())

        if not node_words:
            return 0.0

        # Jaccard similarity
        intersection = len(query_words & node_words)
        union = len(query_words | node_words)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Boost score for exact phrase matches
        if any(word in node.content.lower() for word in query_words):
            jaccard *= 1.2

        return min(1.0, jaccard)

    def get_recall_statistics(self) -> Dict[str, float]:
        """Get statistics about recall performance."""
        all_nodes = self.storage.get_all_nodes()

        total_nodes = len(all_nodes)
        active_nodes = len(
            [n for n in all_nodes if n.activation >= self.activation_threshold]
        )

        if total_nodes == 0:
            return {
                "total_nodes": 0,
                "active_nodes": 0,
                "activation_ratio": 0.0,
                "average_activation": 0.0,
            }

        avg_activation = sum(node.activation for node in all_nodes) / total_nodes
        activation_ratio = active_nodes / total_nodes

        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "activation_ratio": activation_ratio,
            "average_activation": avg_activation,
        }
