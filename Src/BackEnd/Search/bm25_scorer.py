from typing import List, Dict, Any
import logging
from rank_bm25 import BM25Okapi
from BackEnd.Search.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class BM25Scorer:
    """
    Implements BM25 scoring using the rank_bm25 package
    """
    
    def __init__(self):
        """Initialize BM25Scorer"""
        self.text_processor = TextProcessor()
        
    def _prepare_document(self, card: Dict[str, Any]) -> List[str]:
        """
        Prepare card document for BM25 scoring by combining and weighting relevant fields
        
        Args:
            card: Card document
            
        Returns:
            List of tokenized words from card text
        """
        # Get text from each field and normalize
        name = self.text_processor.normalize_text(card.get('name', ''))
        text = self.text_processor.normalize_text(card.get('text', ''))
        type_text = self.text_processor.normalize_text(card.get('type', ''))
        flavor_text = self.text_processor.normalize_text(card.get('flavorText', ''))
        subtypes = ' '.join(card.get('subtypes', []))
        subtypes = self.text_processor.normalize_text(subtypes)
        
        # Extract words from each field
        name_words = self.text_processor.extract_words(name, 'name')
        text_words = self.text_processor.extract_words(text, 'text')
        type_words = self.text_processor.extract_words(type_text, 'type')
        flavor_words = self.text_processor.extract_words(flavor_text, 'flavorText')
        subtype_words = self.text_processor.extract_words(subtypes, 'subtypes')
        
        # Weight words according to field importance
        weighted_words = []
        weighted_words.extend(name_words * int(self.text_processor.get_field_weight('name') * 10))
        weighted_words.extend(text_words * int(self.text_processor.get_field_weight('text') * 10))
        weighted_words.extend(type_words * int(self.text_processor.get_field_weight('type') * 10))
        weighted_words.extend(subtype_words * int(self.text_processor.get_field_weight('subtypes') * 10))
        weighted_words.extend(flavor_words * int(self.text_processor.get_field_weight('flavorText') * 10))
        
        return weighted_words
        
    def score_cards(self, query: str, cards: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """
        Score cards using BM25 algorithm and return top K results
        
        Args:
            query: Expanded user query
            cards: List of card documents to score
            k: Number of top cards to return
            
        Returns:
            List of top K cards sorted by BM25 score
        """
        if not cards:
            logger.warning("No cards provided for scoring")
            return []
            
        logger.info(f"Scoring {len(cards)} cards with query: {query}")
        
        # Normalize query and prepare corpus
        normalized_query = self.text_processor.normalize_text(query)
        query_terms = self.text_processor.extract_words(normalized_query)
        
        # Prepare corpus
        corpus = [self._prepare_document(card) for card in cards]
        
        # Initialize BM25 model
        bm25 = BM25Okapi(corpus)
        
        # Get scores for all documents
        scores = bm25.get_scores(query_terms)
        
        # Create (card, score) pairs and sort by score
        card_scores = list(zip(cards, scores))
        card_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K cards
        top_k_cards = [card for card, _ in card_scores[:k]]
        
        logger.info(f"Returning top {len(top_k_cards)} cards")
        return top_k_cards
