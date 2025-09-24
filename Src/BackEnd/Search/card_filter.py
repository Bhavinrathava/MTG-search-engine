import time
import logging
from typing import List, Dict, Any
from BackEnd.Search.database_connection import DatabaseConnection
from BackEnd.Search.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class CardFilter:
    """Handles card filtering and scoring logic"""
    
    def __init__(self):
        self.db_connection = DatabaseConnection()
        self.text_processor = TextProcessor()
    
    def inverted_search(self, expanded_text: str) -> List[str]:
        """
        Enhanced inverted search with field weighting and importance scoring
        
        Args:
            expanded_text: The expanded query text to search with
            
        Returns:
            List of card UUIDs sorted by relevance
        """
        logger.info("Starting inverted search with expanded text: %s", expanded_text)
        start_time = time.time()
        
        db = self.db_connection.get_connection()
        if db is None:  # Changed from 'if not db' to 'if db is None'
            logger.error("Could not connect to database")
            return []
            
        index_collection = db['InvertedIndex']
        
        # Extract words from expanded text
        word_start = time.time()
        words = self.text_processor.extract_words(expanded_text)
        word_time = time.time() - word_start
        logger.info("Word extraction completed in %.3fs - Found %d words", word_time, len(words))
        
        # Store word scores and their associated card IDs
        word_scores = {}
        card_scores = {}
        logger.debug("Processing words and calculating scores...")
        
        # Process words and calculate scores
        scoring_start = time.time()
        for word in words:
            word_lower = word.lower().strip()
            if not word_lower:
                continue
                
            result = index_collection.find_one({
                '_id': word_lower,
                'doc_frequency': {'$lt': 0.1}  # Filter out very common words
            })
            
            if not result:
                logger.debug("No index entry found for word: %s", word_lower)
                continue
            
            logger.debug("Processing word '%s' with importance %.2f", 
                        word_lower, result.get('importance_score', 1.0))
            
            # Calculate word importance
            importance = result.get('importance_score', 1.0)
            
            # Get field-specific counts
            field_counts = result.get('field_counts', {})
            
            # Calculate weighted score for this word
            word_score = importance * sum(
                count * self.text_processor.get_field_weight(field)
                for field, count in field_counts.items()
            )
            
            word_scores[word_lower] = word_score
            
            # Update card scores
            for card_id in result.get('documents', []):
                if card_id not in card_scores:
                    card_scores[card_id] = 0
                card_scores[card_id] += word_score
        
        scoring_time = time.time() - scoring_start
        logger.info("Word processing & scoring completed in %.3fs", scoring_time)
        logger.info("Found scores for %d words and %d cards", len(word_scores), len(card_scores))
        
        # Sort cards by score
        sort_start = time.time()
        sorted_cards = sorted(
            card_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        sort_time = time.time() - sort_start
        total_time = time.time() - start_time
        
        logger.info("Card sorting completed in %.3fs", sort_time)
        logger.info("Total inverted search time: %.3fs", total_time)
        logger.info("Found %d matching cards", len(sorted_cards))
        
        return [card_id for card_id, _ in sorted_cards]
    
    def filter_cards(self, expanded_text: str, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter cards based on inverted search and query parameters
        
        Args:
            expanded_text: The expanded query text
            query_params: MongoDB query parameters
            
        Returns:
            List of filtered card documents
        """
        logger.info("\nStarting card filtering process")
        logger.info("Query parameters: %s", query_params)
        start_time = time.time()
        
        # Run inverted search
        parallel_start = time.time()
        card_ids = self.inverted_search(expanded_text)
        inverted_search_time = time.time() - parallel_start
        logger.info("Inverted search completed in %.3fs - Found %d card IDs", 
                   inverted_search_time, len(card_ids))
        
        # Add card IDs to query parameters
        query_params['uuid'] = {'$in': card_ids}
        logger.debug("Updated query parameters with card IDs")
        
        # MongoDB query execution
        mongo_start = time.time()
        db = self.db_connection.get_connection()
        if db is None:  # Changed from 'if not db' to 'if db is None'
            logger.error("Could not connect to database for card filtering")
            return []
            
        collection = db['Cards']
        results = collection.find(query_params)
        card_subset = list(results)
        
        # If no results, try fallback with just card IDs
        if len(card_subset) == 0:
            logger.warning("No results with full parameters, trying fallback with just card IDs")
            fallback_params = {'uuid': {'$in': card_ids}}
            fallback_results = collection.find(fallback_params)
            card_subset = list(fallback_results)
            logger.info("Fallback search found %d cards", len(card_subset))
        
        mongo_time = time.time() - mongo_start
        total_time = time.time() - start_time
        
        logger.info("MongoDB query completed in %.3fs", mongo_time)
        logger.info("Total filtering time: %.3fs", total_time)
        logger.info("Final result count: %d cards", len(card_subset))
        
        return card_subset
    
    def get_card_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """
        Retrieve full card data by UUID from MongoDB
        
        Args:
            uuid: The UUID of the card to retrieve
            
        Returns:
            Card document or None if not found
        """
        logger.debug("Retrieving card with UUID: %s", uuid)
        try:
            db = self.db_connection.get_connection()
            if db is None:  # Changed from 'if not db' to 'if db is None'
                logger.error("Database connection failed while retrieving card")
                return None
                
            collection = db['Cards']
            card = collection.find_one({'uuid': uuid})
            if card:
                logger.debug("Successfully retrieved card: %s", card.get('name', 'Unknown'))
            else:
                logger.warning("No card found with UUID: %s", uuid)
            return card
            
        except Exception as e:
            logger.error("Error retrieving card by UUID %s: %s", uuid, e)
            return None
    
    def deduplicate_cards(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate cards based on card name
        
        Args:
            cards: List of card documents to deduplicate
            
        Returns:
            List of deduplicated card documents
        """
        logger.info("Deduplicating %d cards", len(cards))
        final_list = []
        visited_names = set()

        for card in cards:
            card_name = card.get('name', 'Unknown')
            
            if card_name in visited_names:
                logger.debug("Skipping duplicate card: %s", card_name)
                continue
            visited_names.add(card_name)
            final_list.append(card)

        logger.info("Deduplication complete - Reduced from %d to %d unique cards", 
                   len(cards), len(final_list))
        return final_list
