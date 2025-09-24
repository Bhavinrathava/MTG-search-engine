import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Handles embedding-related operations for card search"""
    
    _instance = None
    _model = None
    _model_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.model_name = 'all-MiniLM-L6-v2'
    
    def _get_model(self) -> Optional[SentenceTransformer]:
        """Get or initialize the embedding model"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:  # Double-check pattern
                    logger.info("Initializing sentence transformer model: %s", self.model_name)
                    try:
                        self._model = SentenceTransformer(self.model_name)
                        logger.info("Model initialization successful")
                    except Exception as e:
                        logger.error("Error loading embedding model: %s", e)
                        return None
        return self._model
    
    def get_card_embedding(self, card: Dict[str, Any]) -> Optional[List[float]]:
        """
        Get embedding for a card, either from stored embedding or by generating new one
        
        Args:
            card: Card document containing text and possibly stored embedding
            
        Returns:
            List of embedding values or None if embedding cannot be generated
        """
        card_name = card.get('name', 'Unknown Card')
        logger.debug("Getting embedding for card: %s", card_name)
        
        # Check for stored embedding
        if 'embedding' in card and card['embedding']:
            logger.debug("Using stored embedding for card: %s", card_name)
            return card['embedding']
        
        # Generate new embedding from original text
        original_text = card.get('originalText', '')
        if not original_text or not original_text.strip():
            logger.warning("No original text found for card: %s", card_name)
            return None
            
        try:
            model = self._get_model()
            if model:
                logger.info("Generating new embedding for card: %s", card_name)
                embedding = model.encode(original_text.strip()).tolist()
                logger.debug("Successfully generated embedding for card: %s", card_name)
                return embedding
        except Exception as e:
            logger.error("Error generating card embedding for %s: %s", card_name, e)
        
        return None
    
    def process_cards(self, cards: List[Dict[str, Any]], query_embedding: Union[List[float], np.ndarray], k: int) -> List[Dict[str, Any]]:
        """
        Process cards using embeddings and return top K results
        
        Args:
            cards: List of card documents to process
            query_embedding: Embedding of the query (can be List[float] or numpy array)
            k: Number of top results to return
            
        Returns:
            List of top K cards sorted by similarity
        """
        logger.info("Processing %d cards with embeddings to find top %d matches", len(cards), k)
        
        if not cards:
            logger.warning("No cards provided")
            return []
            
        if query_embedding is None:
            logger.warning("No query embedding provided")
            return []
            
        try:
            # Ensure query_embedding is a numpy array
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            elif not isinstance(query_embedding, np.ndarray):
                logger.error("Invalid query embedding type: %s", type(query_embedding))
                return []
                
            # Process embeddings
            card_embeddings = []
            valid_cards = []
            logger.info("Generating embeddings for cards...")
            
            for card in cards:
                embedding = self.get_card_embedding(card)
                if embedding is not None:  # Explicit None check
                    try:
                        # Convert embedding to numpy array for validation
                        embedding_array = np.array(embedding)
                        if embedding_array.size > 0:  # Check if embedding is not empty
                            card_embeddings.append(embedding_array)
                            valid_cards.append(card)
                        else:
                            logger.warning("Empty embedding for card: %s", card.get('name', 'Unknown'))
                    except Exception as e:
                        logger.error("Error processing embedding for card %s: %s", 
                                   card.get('name', 'Unknown'), e)
            
            if not card_embeddings:
                logger.warning("No valid embeddings generated, returning first %d cards", k)
                return cards[:k]  # Return first K cards if no embeddings available
            
            # Calculate similarities
            try:
                logger.info("Calculating similarity scores for %d cards", len(card_embeddings))
                card_embeddings_array = np.stack(card_embeddings)  # Stack instead of array for consistent dimensions
                query_embedding_array = query_embedding.reshape(1, -1)
                
                # Ensure dimensions match
                if card_embeddings_array.shape[1] != query_embedding_array.shape[1]:
                    logger.error("Embedding dimension mismatch: query=%s, cards=%s", 
                               query_embedding_array.shape, card_embeddings_array.shape)
                    return cards[:k]
                
                similarities = cosine_similarity(query_embedding_array, card_embeddings_array)[0]
                similarities_list = similarities.tolist()  # Convert to list to avoid numpy comparison issues
                
                # Create list of (index, similarity) tuples and sort
                indexed_similarities = list(enumerate(similarities_list))
                indexed_similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top K indices
                top_indices = [idx for idx, _ in indexed_similarities[:k]]
                
                logger.info("Found top %d matches with similarity scores", k)
                for idx, (card_idx, similarity) in enumerate(indexed_similarities[:k], 1):
                    card_name = valid_cards[card_idx].get('name', 'Unknown')
                    logger.info("Match %d: %s (similarity: %.3f)", idx, card_name, similarity)
                
                # Return top K cards
                return [valid_cards[i] for i in top_indices]
                
            except Exception as e:
                logger.error("Error calculating similarities: %s", e)
                return valid_cards[:k]
                
        except Exception as e:
            logger.error("Error in process_cards: %s", e)
            return cards[:k]
    
    def fast_search(self, cards: List[Dict[str, Any]], query_embedding: Union[List[float], np.ndarray], 
                   similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Perform fast search using embeddings with early stopping
        
        Args:
            cards: List of card documents to search
            query_embedding: Embedding of the query
            similarity_threshold: Threshold for considering a match
            
        Returns:
            List of matching cards
        """
        try:
            # Ensure query_embedding is a numpy array
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            query_embedding_array = query_embedding.reshape(1, -1)
            
            results = []
            for card in cards:
                embedding = self.get_card_embedding(card)
                if embedding is not None:
                    try:
                        card_embedding = np.array(embedding).reshape(1, -1)
                        if card_embedding.shape[1] == query_embedding_array.shape[1]:
                            similarity = float(cosine_similarity(
                                query_embedding_array, card_embedding
                            )[0][0])
                            
                            if similarity >= similarity_threshold:
                                results.append((card, similarity))
                                
                                # Early stopping if we found a very good match
                                if similarity > 0.95:
                                    break
                        else:
                            logger.warning("Embedding dimension mismatch for card: %s", 
                                         card.get('name', 'Unknown'))
                    except Exception as e:
                        logger.error("Error processing card %s in fast search: %s", 
                                   card.get('name', 'Unknown'), e)
            
            # Sort by similarity and return cards
            results.sort(key=lambda x: x[1], reverse=True)
            return [card for card, _ in results]
            
        except Exception as e:
            logger.error("Error in fast_search: %s", e)
            return []
