import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Set, Tuple
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

class Query:
    def __init__(self, queryText):
        self.originalText = queryText
        self.expandedQueryWords = []
        self.cleanedWords = []
        self.expandedText = ""
        self.embedding = None
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # MTG-specific stop words to add
        mtg_stop_words = {'card', 'cards', 'magic', 'gathering', 'mtg', 'deck', 'play'}
        self.stop_words.update(mtg_stop_words)
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.warning(f"Could not load embedding model: {e}")
            self.embedding_model = None
        
        # MTG-specific synonym dictionary for better results
        self.mtg_synonyms = {
            # Damage related
            'damage': ['harm', 'hurt', 'wound', 'injury', 'destruction'],
            'destroy': ['eliminate', 'remove', 'annihilate', 'obliterate'],
            'deal': ['cause', 'inflict', 'give'],
            
            # Creature related
            'creature': ['being', 'monster', 'beast', 'entity'],
            'summon': ['call', 'invoke', 'bring forth'],
            'flying': ['airborne', 'aerial'],
            'trample': ['overrun', 'crush'],
            
            # Spells and abilities
            'spell': ['magic', 'incantation', 'enchantment'],
            'instant': ['immediate', 'quick', 'fast'],
            'sorcery': ['magic', 'spell', 'enchantment'],
            'artifact': ['item', 'object', 'device', 'tool'],
            'enchantment': ['charm', 'magic', 'spell'],
            
            # Colors
            'red': ['crimson', 'scarlet', 'fire'],
            'blue': ['azure', 'water', 'sea'],
            'green': ['forest', 'nature', 'life'],
            'white': ['light', 'pure', 'holy'],
            'black': ['dark', 'death', 'shadow'],
            
            # Actions
            'draw': ['take', 'get', 'obtain'],
            'discard': ['throw away', 'dispose'],
            'exile': ['banish', 'remove'],
            'counter': ['negate', 'cancel', 'stop'],
            
            # Power/Toughness
            'power': ['strength', 'attack', 'force'],
            'toughness': ['defense', 'durability', 'health'],
            
            # Common terms
            'target': ['choose', 'select', 'pick'],
            'battlefield': ['field', 'play area', 'board'],
            'graveyard': ['discard pile', 'cemetery'],
            'library': ['deck'],
            'hand': ['cards in hand'],
        }
    
    def clean_text(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove empty tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def get_wordnet_pos(self, word: str) -> str:
        """Get WordNet POS tag for better lemmatization"""
        try:
            import nltk
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN
    
    def removeStopWords(self):
        """Remove stop words from the query"""
        tokens = self.clean_text(self.originalText)
        self.cleanedWords = [word for word in tokens if word.lower() not in self.stop_words and len(word) > 1]
    
    def get_wordnet_synonyms(self, word: str, max_synonyms: int = 3) -> Set[str]:
        """Get synonyms from WordNet"""
        synonyms = set()
        
        # Get lemmatized form
        lemmatized_word = self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
        
        # Get synsets for the word
        synsets = wordnet.synsets(lemmatized_word)
        
        for synset in synsets[:2]:  # Limit to first 2 synsets for relevance
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym) > 1:
                    synonyms.add(synonym)
                    if len(synonyms) >= max_synonyms:
                        break
            if len(synonyms) >= max_synonyms:
                break
        
        return synonyms
    
    def get_mtg_synonyms(self, word: str) -> Set[str]:
        """Get MTG-specific synonyms"""
        return set(self.mtg_synonyms.get(word.lower(), []))
    
    def addSynonyms(self):
        """Add synonyms to expand the query"""
        self.expandedQueryWords = self.cleanedWords.copy()
        
        for word in self.cleanedWords:
            # Get MTG-specific synonyms first (more relevant)
            mtg_syns = self.get_mtg_synonyms(word)
            self.expandedQueryWords.extend(list(mtg_syns))
            
            # Get WordNet synonyms if we don't have enough MTG synonyms
            if len(mtg_syns) < 2:
                wordnet_syns = self.get_wordnet_synonyms(word, max_synonyms=2)
                self.expandedQueryWords.extend(list(wordnet_syns))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in self.expandedQueryWords:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        self.expandedQueryWords = unique_words
    
    def create_expanded_text(self) -> str:
        """Create expanded query text from expanded words"""
        self.expandedText = ' '.join(self.expandedQueryWords)
        return self.expandedText
    
    def generate_embedding(self) -> np.ndarray:
        """Generate embedding for the expanded query"""
        if self.embedding_model is None:
            logging.warning("Embedding model not available")
            return None
        
        if not self.expandedText:
            self.create_expanded_text()
        
        try:
            # Generate embedding for the expanded text
            self.embedding = self.embedding_model.encode(self.expandedText)
            return self.embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def expandQuery(self) -> Tuple[str, np.ndarray]:
        """
        Complete query expansion pipeline
        Returns: (expanded_query_string, embedding_vector)
        """
        # Step 1: Remove stop words
        self.removeStopWords()
        
        # Step 2: Add synonyms
        self.addSynonyms()
        
        # Step 3: Create expanded text
        expanded_text = self.create_expanded_text()
        
        # Step 4: Generate embedding
        embedding = self.generate_embedding()
        
        return expanded_text, embedding
    
    def get_expansion_details(self) -> dict:
        """Get detailed information about the query expansion"""
        return {
            'original_text': self.originalText,
            'cleaned_words': self.cleanedWords,
            'expanded_words': self.expandedQueryWords,
            'expanded_text': self.expandedText,
            'embedding_shape': self.embedding.shape if self.embedding is not None else None,
            'expansion_ratio': len(self.expandedQueryWords) / len(self.cleanedWords) if self.cleanedWords else 0
        }
    
    def find_similar_queries(self, other_queries: List['Query'], top_k: int = 5) -> List[Tuple['Query', float]]:
        """
        Find similar queries using cosine similarity of embeddings
        """
        if self.embedding is None:
            return []
        
        similarities = []
        for other_query in other_queries:
            if other_query.embedding is not None:
                # Compute cosine similarity
                dot_product = np.dot(self.embedding, other_query.embedding)
                norm_a = np.linalg.norm(self.embedding)
                norm_b = np.linalg.norm(other_query.embedding)
                
                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    similarities.append((other_query, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# Example usage and testing
def test_query_expansion():
    """Test the Query class with various MTG-related queries"""
    
    test_queries = [
        "lightning bolt red instant",
        "flying creature blue",
        "destroy target artifact", 
        "draw cards from library",
        "powerful dragon with trample"
    ]
    
    expanded_queries = []
    
    for query_text in test_queries:
        print(f"\n{'='*50}")
        print(f"Original Query: '{query_text}'")
        
        # Create and expand query
        query = Query(query_text)
        expanded_text, embedding = query.expandQuery()
        
        # Get details
        details = query.get_expansion_details()
        
        print(f"Cleaned Words: {details['cleaned_words']}")
        print(f"Expanded Words: {details['expanded_words']}")
        print(f"Expanded Text: '{expanded_text}'")
        print(f"Expansion Ratio: {details['expansion_ratio']:.2f}")
        
        if embedding is not None:
            print(f"Embedding Shape: {embedding.shape}")
            print(f"Embedding Sample: [{embedding[:5]}...]")
        else:
            print("Embedding: Not generated")
        
        expanded_queries.append(query)
    
    # Test similarity between queries
    if len(expanded_queries) > 1 and all(q.embedding is not None for q in expanded_queries):
        print(f"\n{'='*50}")
        print("Query Similarity Test:")
        
        base_query = expanded_queries[0]
        similar_queries = base_query.find_similar_queries(expanded_queries[1:], top_k=3)
        
        print(f"Queries similar to '{base_query.originalText}':")
        for similar_query, similarity in similar_queries:
            print(f"  - '{similar_query.originalText}' (similarity: {similarity:.3f})")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import sentence_transformers
    except ImportError:
        print("Please install sentence-transformers: pip install sentence-transformers")
    
    # Run tests
    test_query_expansion()
