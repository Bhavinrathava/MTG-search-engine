import json
import re
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher

class TrieNode:
    """Node for the TRIE data structure"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_word: bool = False
        self.card_names: List[str] = []
        self.uuid: Optional[str] = None

class CardNameTrie:
    """TRIE implementation optimized for MTG card name searches"""
    
    def __init__(self, name_to_uuid_file: str = 'card_name_to_uuid.json'):
        self.root = TrieNode()
        self.name_to_uuid: Dict[str, str] = {}
        self.uuid_to_name: Dict[str, str] = {}
        self.cache: Dict[str, List[Tuple[str, str, float]]] = {}
        self.cache_size = 1000  # Maximum number of items to cache
        self.load_card_mappings(name_to_uuid_file)
        self.build_trie()
    
    def load_card_mappings(self, file_path: str):
        """Load the card name to UUID mappings"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'name_to_uuid' in data:
                self.name_to_uuid = data['name_to_uuid']
                self.uuid_to_name = data.get('uuid_to_name', {})
            elif 'card_mappings' in data:
                self.name_to_uuid = data['card_mappings']
            else:
                self.name_to_uuid = data
            
            if not self.uuid_to_name:
                self.uuid_to_name = {uuid: name for name, uuid in self.name_to_uuid.items()}
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading card mappings: {e}")
            self.name_to_uuid = {}
            self.uuid_to_name = {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def build_trie(self):
        """Build the TRIE from card names"""
        for card_name, uuid in self.name_to_uuid.items():
            self.insert(card_name, uuid)
    
    def insert(self, card_name: str, uuid: str):
        """Insert a card name into the TRIE"""
        normalized_name = self.normalize_text(card_name)
        node = self.root
        
        for char in normalized_name:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_word = True
        node.card_names.append(card_name)
        node.uuid = uuid
    
    def exact_search(self, query: str) -> Optional[str]:
        """Find exact match for a card name"""
        normalized_query = self.normalize_text(query)
        node = self.root
        
        for char in normalized_query:
            if char not in node.children:
                return None
            node = node.children[char]
        
        if node.is_end_word:
            return node.uuid
        return None
    
    def prefix_search(self, prefix: str, max_results: int = 10) -> List[Tuple[str, str]]:
        """Find all card names that start with the given prefix"""
        normalized_prefix = self.normalize_text(prefix)
        node = self.root
        
        for char in normalized_prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        self._collect_words(node, results, max_results)
        return results[:max_results]
    
    def _collect_words(self, node: TrieNode, results: List[Tuple[str, str]], max_results: int):
        """Recursively collect all words from a node"""
        if len(results) >= max_results:
            return
        
        if node.is_end_word:
            for card_name in node.card_names:
                results.append((card_name, node.uuid))
                if len(results) >= max_results:
                    return
        
        for child in node.children.values():
            self._collect_words(child, results, max_results)
            if len(results) >= max_results:
                return
    
    def fuzzy_search(self, query: str, max_distance: int = 2, max_results: int = 10) -> List[Tuple[str, str, float]]:
        """Find card names similar to the query using similarity matching with caching"""
        normalized_query = self.normalize_text(query)
        
        # Check cache first
        cache_key = f"fuzzy_{normalized_query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Split query into words for partial matching
        query_words = normalized_query.split()
        results = []
        
        # First try exact word matches
        exact_matches = []
        for card_name, uuid in self.name_to_uuid.items():
            normalized_card = self.normalize_text(card_name)
            if any(word in normalized_card for word in query_words):
                exact_matches.append((card_name, uuid))
        
        # Calculate similarity only for exact matches first
        for card_name, uuid in exact_matches:
            normalized_card = self.normalize_text(card_name)
            similarity = SequenceMatcher(None, normalized_query, normalized_card).ratio()
            if similarity > 0.7:
                results.append((card_name, uuid, similarity))
        
        # If we don't have enough results, try fuzzy matching on remaining cards
        if len(results) < max_results:
            remaining_slots = max_results - len(results)
            for card_name, uuid in self.name_to_uuid.items():
                if (card_name, uuid) not in exact_matches:
                    normalized_card = self.normalize_text(card_name)
                    if len(normalized_card) > 2 and abs(len(normalized_card) - len(normalized_query)) <= max_distance:
                        similarity = SequenceMatcher(None, normalized_query, normalized_card).ratio()
                        if similarity > 0.6:
                            results.append((card_name, uuid, similarity))
        
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:max_results]
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = results
        
        return results
    
    def is_likely_card_name(self, query: str) -> bool:
        """Enhanced detection of whether a query is likely a card name"""
        if not query or len(query.strip()) == 0:
            return False
            
        query = query.strip()
        words = query.split()
        
        if len(words) > 8:
            return False
        
        descriptive_words = ['with', 'that', 'has', 'deals', 'damage', 'mana', 'when', 'if', 
                           'creature', 'instant', 'sorcery', 'enchantment', 'artifact', 'target']
        if sum(1 for word in descriptive_words if word in query.lower()) >= 2:
            return False
        
        exact_uuid = self.exact_search(query)
        if exact_uuid:
            return True
        
        fuzzy_results = self.fuzzy_search(query, max_results=1)
        if fuzzy_results and fuzzy_results[0][2] > 0.8:
            return True
        
        prefix_results = self.prefix_search(query, max_results=1)
        if prefix_results:
            return True
        
        return False
