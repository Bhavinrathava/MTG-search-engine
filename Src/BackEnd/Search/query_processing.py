import logging
from typing import List, Dict, Any, Optional
from BackEnd.Search.queryExpansion import Query
from BackEnd.Search.database_connection import DatabaseConnection
from BackEnd.Search.text_processor import TextProcessor
from BackEnd.Search.gpt_interface import GPTInterface
from BackEnd.Search.card_name_trie import CardNameTrie
from BackEnd.Search.query_parameters import QueryParameters
from BackEnd.Search.card_filter import CardFilter
from BackEnd.Search.embedding_processor import EmbeddingProcessor
from BackEnd.Search.bm25_scorer import BM25Scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryProcessing:
    """Main class that orchestrates the MTG card search process"""
    
    def __init__(self, query_text: str):
        logger.info("Initializing QueryProcessing with query: %s", query_text)
        
        # Initialize all components
        self.original_query_text = query_text
        self.query = Query(query_text)
        logger.info("Expanding query...")
        self.expanded_text, self.query_embedding = self.query.expandQuery()
        logger.info("Query expanded to: %s", self.expanded_text)
        
        # Initialize component classes
        logger.info("Initializing component classes...")
        self.db_connection = DatabaseConnection()
        self.text_processor = TextProcessor()
        self.gpt_interface = GPTInterface()
        self.card_trie = CardNameTrie('card_name_to_uuid.json')
        self.query_params = QueryParameters()
        self.card_filter = CardFilter()
        self.embedding_processor = EmbeddingProcessor()
        self.bm25_scorer = BM25Scorer()
        
        # Track component availability
        self.trie_available = hasattr(self.card_trie, 'name_to_uuid') and bool(self.card_trie.name_to_uuid)
        logger.info("TRIE search available: %s", self.trie_available)
    
    def is_likely_card_name(self, query: str) -> bool:
        """Check if the query is likely a card name"""
        if not self.trie_available:
            words = query.split()
            return len(words) <= 6 and not any(
                desc_word in query.lower() for desc_word in 
                ['with', 'that', 'deals', 'damage', 'mana', 'creature', 'instant']
            )
        
        return self.card_trie.is_likely_card_name(query)
    
    def fast_search(self) -> List[Dict[str, Any]]:
        """Perform fast TRIE-based search for card names"""
        logger.info("Starting fast TRIE-based search")
        
        if not self.trie_available or not self.original_query_text.strip():
            logger.info("Fast search skipped - TRIE not available or empty query")
            return []
        
        results = []
        query = self.original_query_text.strip()
        logger.info("Performing fast search for query: %s", query)
        
        # Try exact match first
        exact_uuid = self.card_trie.exact_search(query)
        if exact_uuid:
            card_data = self.card_filter.get_card_by_uuid(exact_uuid)
            if card_data:
                return [card_data]
        
        # Try prefix search
        prefix_results = self.card_trie.prefix_search(query, max_results=5)
        if prefix_results:
            for card_name, uuid in prefix_results:
                card_data = self.card_filter.get_card_by_uuid(uuid)
                if card_data:
                    results.append(card_data)
        
        # Try fuzzy search if needed
        if not results:
            fuzzy_results = self.card_trie.fuzzy_search(query, max_distance=2, max_results=5)
            if fuzzy_results:
                for card_name, uuid, similarity in fuzzy_results:
                    if similarity > 0.6:
                        card_data = self.card_filter.get_card_by_uuid(uuid)
                        if card_data:
                            results.append(card_data)
        
        return results
    
    def findTopK(self, k: int) -> List[Dict[str, Any]]:
        """
        Find top K matching cards for the query
        
        Args:
            k: Number of results to return
            
        Returns:
            List of top K matching card documents
        """
        logger.info("Finding top %d matches for query", k)
        
        # Try fast search first for likely card names
        if len(self.original_query_text.split()) < 6 and self.is_likely_card_name(self.original_query_text):
            logger.info("Query looks like a card name, attempting fast search")
            fast_results = self.fast_search()
            if len(fast_results) >= 1:
                logger.info("Fast search successful - found %d results", len(fast_results))
                return fast_results[:k]
            logger.info("Fast search yielded no results, falling back to full search")
        
        # Generate query parameters
        logger.info("Generating query parameters")
        query_params = self.query_params.generate_parameters(
            self.original_query_text,
            self.expanded_text
        )
        logger.info("Query parameters generated")
        
        # Filter cards
        logger.info("Filtering cards based on parameters")
        card_subset = self.card_filter.filter_cards(
            self.expanded_text,
            query_params
        )
        logger.info("Initial filtering complete - found %d cards", len(card_subset) if card_subset else 0)
        
        if not card_subset:
            logger.warning("No cards found after filtering")
            return []
            
        # Process with BM25 scoring
        logger.info("Processing %d cards with BM25 scoring", len(card_subset))
        bm25_results = self.bm25_scorer.score_cards(
            self.expanded_text,
            card_subset,
            k=min(k * 2, len(card_subset))  # Get 2x cards for embedding reranking
        )
        logger.info("BM25 scoring complete - selected %d cards", len(bm25_results))
        
        if self.query_embedding is None:
            logger.info("No query embedding available, returning BM25 results")
            return bm25_results[:k]
        
        # Process with embeddings for final ranking
        logger.info("Processing %d cards with embeddings", len(bm25_results))
        top_cards = self.embedding_processor.process_cards(
            bm25_results,
            self.query_embedding,
            k
        )
        logger.info("Embedding processing complete - found %d matches", len(top_cards))
        
        # Deduplicate results
        final_results = self.card_filter.deduplicate_cards(top_cards)
        logger.info("Deduplication complete - returning %d unique cards", len(final_results))
        return final_results
    
    def get_card_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get full card data by UUID"""
        return self.card_filter.get_card_by_uuid(uuid)
