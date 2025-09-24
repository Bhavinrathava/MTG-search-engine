from queryExpansion import Query
from pymongo import MongoClient
import json 
import re
import time
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global shared resources
_mongo_client = None
_mongo_db = None
_embedding_model = None
_model_lock = threading.Lock()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

# MTG-specific stop words that are too common to be useful for indexing
MTG_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'your', 'you', 'this', 'that',
    'its', 'is', 'at', 'if', 'in', 'into', 'of', 'from', 'for', 'with',
    'creature', 'spell', 'card', 'ability', 'target', 'effect', 'combat',
    'damage', 'mana', 'cost', 'tap', 'untap', 'player', 'opponent',
    'battlefield', 'graveyard', 'hand', 'library', 'exile', 'stack',
    'beginning', 'end', 'step', 'phase', 'turn', 'game', 'cast', 'play',
    'control', 'controlled', 'controller', 'owner', 'put', 'add', 'remove',
    'counter', 'each', 'all', 'any', 'every', 'whenever', 'may', 'can',
    'until', 'during'
}

# Field importance weights for scoring
FIELD_WEIGHTS = {
    'name': 1.0,       # Highest weight for card name
    'type': 0.8,       # High weight for card type
    'subtypes': 0.7,   # Important for tribal/theme
    'text': 0.6,       # Card rules text
    'flavorText': 0.3  # Lowest weight for flavor text
}

# Global connection pool - shared across all instances
_mongo_client = None
_mongo_db = None

def get_mongo_connection():
    """Get or create a shared MongoDB connection"""
    global _mongo_client, _mongo_db
    
    if _mongo_client is None:
        uri = "mongodb+srv://bhavinmongocluster.5t6smyb.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster"
        
        _mongo_client = MongoClient(
            uri,
            tls=True,
            tlsCertificateKeyFile="Config/X509-cert-3753233507821277243.pem",
            serverSelectionTimeoutMS=30000,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            maxPoolSize=10,
            minPoolSize=2,
            maxIdleTimeMS=30000,
        )
        _mongo_db = _mongo_client['MTGCards']
        _mongo_db.command('ping')
    
    return _mongo_db

def extract_compound_tokens(text):
    """Extract multi-word tokens that should be kept together"""
    compound_tokens = []
    
    patterns = [
        r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Proper names like "Serra Angel"
        r'[A-Z][a-z]+\s+of\s+[A-Z][a-z]+',  # "X of Y" patterns
        r'\w+\s+\w+(?=\s+[-—]\s+)',  # Card names before dash/em-dash
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            token = match.group().lower()
            if not any(word in MTG_STOP_WORDS for word in token.split()):
                compound_tokens.append(token)
    
    return compound_tokens

def extract_words(text, field_name=None):
    """Extract words from text for indexing with field-specific processing"""
    if not text:
        return []
    
    words = []
    
    # Extract compound tokens first
    compound_tokens = extract_compound_tokens(text)
    words.extend(compound_tokens)
    
    # Remove compound tokens from text to avoid double-counting
    for token in compound_tokens:
        text = text.replace(token, '')
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words and filter
    single_words = [
        word.strip() for word in text.split()
        if len(word.strip()) > 2  # Ignore very short words
        and word.strip() not in MTG_STOP_WORDS  # Filter stop words
    ]
    
    words.extend(single_words)
    return words

def call_gpt(prompt: str,
             model: str = "gpt-4o-mini",
             max_tokens: int = 500,
             temperature: float = 0.7) -> str:
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

class TrieNode:
    """Node for the TRIE data structure"""
    def __init__(self):
        self.children = {}
        self.is_end_word = False
        self.card_names = []
        self.uuid = None

class CardNameTrie:
    """TRIE implementation optimized for MTG card name searches"""
    
    def __init__(self, name_to_uuid_file: str = 'card_name_to_uuid.json'):
        self.root = TrieNode()
        self.name_to_uuid = {}
        self.uuid_to_name = {}
        self.cache = {}  # Cache for frequently accessed cards
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
                
        except (FileNotFoundError, json.JSONDecodeError, Exception):
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

class QueryProcessing:
    def __init__(self, queryText):
        self.originalQueryText = queryText
        self.query = Query(queryText)
        self.expandedText, self.queryEmbedding = self.query.expandQuery()

        # MongoDB connection details
        self.connection_config = {
            'uri': "mongodb+srv://bhavinmongocluster.5t6smyb.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster",
            'tls': True,
            'tlsCertificateKeyFile': "Config/X509-cert-3753233507821277243.pem",
            'serverSelectionTimeoutMS': 60000,
            'socketTimeoutMS': 60000,
            'connectTimeoutMS': 60000
        }
        
        # Initialize TRIE for fast card name searches
        try:
            self.trie = CardNameTrie('card_name_to_uuid.json')
            self.trie_available = True
        except Exception:
            self.trie_available = False

    def invertedSearch(self):
        """Enhanced inverted search with field weighting and importance scoring"""
        start_time = time.time()
        
        db = get_mongo_connection()
        index_collection = db['InvertedIndex']
        
        # Extract words from expanded text
        word_start = time.time()
        words = extract_words(self.expandedText)
        print(f"Word Extraction Time: {time.time() - word_start:.3f}s")
        
        # Store word scores and their associated card IDs
        word_scores = {}
        card_scores = {}
        
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
                continue
            
            # Calculate word importance
            importance = result.get('importance_score', 1.0)
            
            # Get field-specific counts
            field_counts = result.get('field_counts', {})
            
            # Calculate weighted score for this word
            word_score = importance * sum(
                count * FIELD_WEIGHTS.get(field, 0.5) 
                for field, count in field_counts.items()
            )
            
            word_scores[word_lower] = word_score
            
            # Update card scores
            for card_id in result.get('documents', []):
                if card_id not in card_scores:
                    card_scores[card_id] = 0
                card_scores[card_id] += word_score
        
        print(f"Word Processing & Scoring Time: {time.time() - scoring_start:.3f}s")
        
        # Sort cards by score
        sort_start = time.time()
        sorted_cards = sorted(
            card_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        print(f"Card Sorting Time: {time.time() - sort_start:.3f}s")
        print(f"Total Inverted Search Time: {time.time() - start_time:.3f}s")
        
        return [card_id for card_id, _ in sorted_cards]
   
    def setup_database(self):
        """Setup MongoDB connection"""
        try:
            self.client = MongoClient(
                self.connection_config['uri'],
                tls=self.connection_config['tls'],
                tlsCertificateKeyFile=self.connection_config['tlsCertificateKeyFile'],
                serverSelectionTimeoutMS=self.connection_config['serverSelectionTimeoutMS'],
                socketTimeoutMS=self.connection_config['socketTimeoutMS'],
                connectTimeoutMS=self.connection_config['connectTimeoutMS']
            )
            
            self.db = self.client['MTGCards']
            self.db.command('ping')
            self.db_connected = True

        except Exception as e:
            self.db_connected = False
            raise
    
    def findQueryParameters(self):
        start_time = time.time()
        
        # Load template
        template_start = time.time()
        cardTemplate = None
        with open('Data/cardTemplate.json', encoding="utf-8") as f:
            cardTemplate = json.load(f)
        print(f"Template Load Time: {time.time() - template_start:.3f}s")

        prompt = f'''
        You are a Magic The Gathering Card Expert System. You will be given a user query where the user is looking for a specific type of card or a specific card. 
        You will be given the following : 
        - Card Object Template explaining what each field means for a card object stored in DB. 
        - Original User Query in String
        - Expanded User Query in String

        The Cards are stored in a MongoDB Collection. The intent is to use collection.find(query) based on user preferences.
        It is okay if the user query is vague, you have to try to convert user query into a qunatifiable query for MongoDB. Is is okay if the MongoDB Query is not Specific. 

        Your task is to return a JSON representing the query object that is to be passed to the MongoDB's collection.find(query) command.  Keep the query parameters to bare essentials. No need to specify exists and other bare-bone query parameters. Just filters are fine. 
        

        Here is the Card Template :

        {cardTemplate}

        Here is the User Query : 

        {self.originalQueryText}

        Here is the expanded Query : 

        {self.expandedText}
        '''

        # Call GPT API
        gpt_start = time.time()
        llmResponse = call_gpt(prompt)
        print(f"GPT API Call Time: {time.time() - gpt_start:.3f}s")
        print(f"Total Query Parameters Time: {time.time() - start_time:.3f}s")
        
        return llmResponse
    
    def filterCards(self):
        print("\nPerformance Analysis:")
        print("-" * 40)
        start_time = time.time()
        
        if not hasattr(self, 'db_connected') or not self.db_connected:
            db_start = time.time()
            self.setup_database()
            print(f"Database Setup Time: {time.time() - db_start:.3f}s")
            
        # Run inverted search and query parameter generation sequentially
        parallel_start = time.time()
        params = self.findQueryParameters()
        cardIDs = self.invertedSearch()
        print(f"Parallel Operations Time: {time.time() - parallel_start:.3f}s")
        params['uuid'] = {'$in': cardIDs}
        
        # MongoDB query execution
        mongo_start = time.time()
        collection = self.db['Cards']
        results = collection.find(params)
        cardSubset = list(results)
        
        if len(cardSubset) == 0:
            fallback_params = {'uuid': {'$in': cardIDs}}
            fallback_results = collection.find(fallback_params)
            cardSubset = list(fallback_results)
        
        print(f"MongoDB Query Time: {time.time() - mongo_start:.3f}s")
        print(f"Total Filter Time: {time.time() - start_time:.3f}s")
        print("-" * 40)
        
        return cardSubset
    
    def get_card_by_uuid(self, uuid: str) -> Optional[Dict]:
        """Retrieve full card data by UUID from MongoDB"""
        try:
            if not hasattr(self, 'db_connected') or not self.db_connected:
                self.setup_database()
            
            collection = self.db['Cards']
            card = collection.find_one({'uuid': uuid})
            return card
            
        except Exception:
            return None
    
    def is_likely_card_name(self, query: str) -> bool:
        """Enhanced detection of whether a query is likely a card name using TRIE"""
        if not self.trie_available or not query or len(query.strip()) == 0:
            words = query.split()
            return len(words) <= 6 and not any(desc_word in query.lower() for desc_word in 
                                             ['with', 'that', 'deals', 'damage', 'mana', 'creature', 'instant'])
        
        query = query.strip()
        words = query.split()
        
        if len(words) > 8:
            return False
        
        descriptive_words = ['with', 'that', 'has', 'deals', 'damage', 'mana', 'when', 'if', 
                           'creature', 'instant', 'sorcery', 'enchantment', 'artifact', 'target']
        if sum(1 for word in descriptive_words if word in query.lower()) >= 2:
            return False
        
        exact_uuid = self.trie.exact_search(query)
        if exact_uuid:
            return True
        
        fuzzy_results = self.trie.fuzzy_search(query, max_results=1)
        if fuzzy_results and fuzzy_results[0][2] > 0.8:
            return True
        
        prefix_results = self.trie.prefix_search(query, max_results=1)
        if prefix_results:
            return True
        
        return False
    
    def fastSearch(self) -> List[Dict]:
        """Fast TRIE-based search for card names"""
        if not self.trie_available or not self.originalQueryText or not self.originalQueryText.strip():
            return []
        
        query = self.originalQueryText.strip()
        results = []
        
        exact_uuid = self.trie.exact_search(query)
        if exact_uuid:
            card_data = self.get_card_by_uuid(exact_uuid)
            if card_data:
                return [card_data]
        
        prefix_results = self.trie.prefix_search(query, max_results=5)
        if prefix_results:
            for card_name, uuid in prefix_results:
                card_data = self.get_card_by_uuid(uuid)
                if card_data:
                    results.append(card_data)
        
        if not results:
            fuzzy_results = self.trie.fuzzy_search(query, max_distance=2, max_results=5)
            if fuzzy_results:
                for card_name, uuid, similarity in fuzzy_results:
                    if similarity > 0.6:
                        card_data = self.get_card_by_uuid(uuid)
                        if card_data:
                            results.append(card_data)
        
        return results
    
    def findTopK(self, k):
        global _embedding_model, _model_lock
        
        start_time = time.time()
        
        # Try fast search first for likely card names
        if len(self.originalQueryText.split()) < 6 and self.is_likely_card_name(self.originalQueryText):
            trie_start = time.time()
            fastResults = self.fastSearch()
            print(f"TRIE Search Time: {time.time() - trie_start:.3f}s")
            if len(fastResults) >= 1:
                return fastResults[:k]

        cardSubsets = self.filterCards()
        
        if not cardSubsets or self.queryEmbedding is None:
            return []
        
        # Load or get cached embedding model
        model_start = time.time()
        try:
            with _model_lock:
                if _embedding_model is None:
                    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding_model = _embedding_model
            print(f"Model Load/Cache Time: {time.time() - model_start:.3f}s")
        except Exception:
            print("Failed to load embedding model, returning filtered results")
            return cardSubsets[:k]
        
        # Process embeddings
        embed_start = time.time()
        card_embeddings = []
        valid_cards = []
        
        for card in cardSubsets:
            embedding = None
            
            if 'embedding' in card and card['embedding']:
                embedding = card['embedding']
            else:
                try:
                    original_text = card.get('originalText', '')
                    if original_text and original_text.strip():
                        embedding = embedding_model.encode(original_text.strip()).tolist()
                except Exception:
                    pass
            
            if embedding:
                card_embeddings.append(embedding)
                valid_cards.append(card)
        
        if not card_embeddings:
            print("No embeddings found, returning filtered results")
            return cardSubsets[:k]
        
        # Calculate similarities
        sim_start = time.time()
        card_embeddings = np.array(card_embeddings)
        query_embedding = np.array(self.queryEmbedding).reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, card_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        topK = [valid_cards[i] for i in top_indices]
        print(f"Similarity Calculation Time: {time.time() - sim_start:.3f}s")
        print(f"Total Embedding Processing Time: {time.time() - embed_start:.3f}s")
        print(f"Total findTopK Time: {time.time() - start_time:.3f}s")
        print("-" * 40)
        
        return topK

    def deduplication(self, cards):
        finalList = []
        visitedNames = set()

        for card in cards: 
            cardName = card.get('name', 'Unknown')
            
            if cardName in visitedNames:
                continue
            visitedNames.add(cardName)
            finalList.append(card)

        return finalList 

def test_complete_pipeline():
    """Test the complete QueryProcessing pipeline with timing and shared connections"""
    print("=" * 60)
    print("MTG QueryProcessing Pipeline Performance Test")
    print("=" * 60)
    
    test_queries = [
        "Lightning Bolt",
        "lightning",
        "Jace, the Mind Sculptor",
        "jace mind",
        "red creature with flying",
        "counterspell",
        "destroy target creature"
    ]
    
    # Initialize shared connection pool once
    print("Initializing connection pool...")
    init_start = time.time()
    get_mongo_connection()
    init_time = time.time() - init_start
    print(f"Connection pool initialized in {init_time:.3f}s\n")
    
    total_time = 0
    
    for test_query in test_queries:
        print(f"Query: '{test_query}'")
        
        start_time = time.time()
        
        try:
            processor = QueryProcessing(test_query)
            
            k = 10
            top_cards = processor.findTopK(k)
            top_cards = processor.deduplication(top_cards)
            
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
            
            print(f"Time: {query_time:.3f}s | Results: {len(top_cards)} unique cards")
            
            for i, card in enumerate(top_cards[:3], 1):
                name = card.get('name', 'Unknown')
                card_type = card.get('type', 'Unknown type')
                mana_cost = card.get('manaCost', 'No mana cost')
                print(f"  {i}. {name} | {card_type} | {mana_cost}")
                    
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
            print(f"Time: {query_time:.3f}s | ERROR: {e}")
        
        print("-" * 40)

    print(f"\nSetup time: {init_time:.3f}s")
    print(f"Total query time: {total_time:.3f}s")
    print(f"Average time per query: {total_time/len(test_queries):.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    test_complete_pipeline()
