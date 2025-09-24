import json
import logging
from typing import Dict, Any
from BackEnd.Search.gpt_interface import GPTInterface

logger = logging.getLogger(__name__)

class QueryParameters:
    """Handles generation of MongoDB query parameters"""
    
    def __init__(self):
        self.gpt_interface = GPTInterface()
        self.card_template = self._load_card_template()
    
    def _load_card_template(self) -> Dict[str, Any]:
        """Load the card template from JSON file"""
        logger.info("Loading card template from Data/cardTemplate.json")
        try:
            with open('Src/BackEnd/Search/Data/cardTemplate.json', encoding="utf-8") as f:
                template = json.load(f)
                logger.info("Card template loaded successfully")
                return template
        except Exception as e:
            logger.error("Error loading card template: %s", e)
            return {}
    
    def generate_parameters(self, original_query: str, expanded_query: str) -> Dict[str, Any]:
        """
        Generate MongoDB query parameters based on the user's query
        
        Args:
            original_query: The original user query
            expanded_query: The expanded version of the query
            
        Returns:
            Dict containing MongoDB query parameters
        """
        logger.info("Generating query parameters")
        logger.debug("Original query: %s", original_query)
        logger.debug("Expanded query: %s", expanded_query)
        
        if not self.card_template:
            logger.warning("Card template not loaded, query parameters may be limited")
            return {}
            
        try:
            # Use GPT to generate query parameters
            logger.info("Requesting GPT to generate query parameters")
            params = self.gpt_interface.generate_query_parameters(
                original_query=original_query,
                expanded_query=expanded_query,
                card_template=self.card_template
            )
            
            # Handle potential errors in GPT response
            if isinstance(params, dict) and 'error' not in params:
                logger.info("Query parameters generated successfully")
                logger.debug("Generated parameters: %s", params)
                return params
            else:
                error_msg = params.get('error', 'Unknown error')
                logger.error("Error in GPT response: %s", error_msg)
                return {}
                
        except Exception as e:
            logger.error("Error generating query parameters: %s", e)
            return {}
    
    def merge_with_card_ids(self, params: Dict[str, Any], card_ids: list) -> Dict[str, Any]:
        """
        Merge query parameters with a list of card IDs
        
        Args:
            params: The base query parameters
            card_ids: List of card IDs to include in the query
            
        Returns:
            Dict containing merged query parameters
        """
        logger.info("Merging query parameters with %d card IDs", len(card_ids))
        
        if not params:
            logger.debug("No base parameters, using only card IDs")
            return {'uuid': {'$in': card_ids}}
            
        merged_params = params.copy()
        merged_params['uuid'] = {'$in': card_ids}
        logger.debug("Merged parameters: %s", merged_params)
        return merged_params
    
    def get_fallback_parameters(self, card_ids: list) -> Dict[str, Any]:
        """
        Get fallback query parameters when main query returns no results
        
        Args:
            card_ids: List of card IDs to include in the query
            
        Returns:
            Dict containing fallback query parameters
        """
        logger.info("Creating fallback parameters with %d card IDs", len(card_ids))
        fallback = {'uuid': {'$in': card_ids}}
        logger.debug("Fallback parameters: %s", fallback)
        return fallback
