import re
from typing import List

class TextProcessor:
    """Handles text processing utilities for MTG card search"""
    
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

    @staticmethod
    def extract_compound_tokens(text: str) -> List[str]:
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
                if not any(word in TextProcessor.MTG_STOP_WORDS for word in token.split()):
                    compound_tokens.append(token)
        
        return compound_tokens

    @staticmethod
    def extract_words(text: str, field_name: str = None) -> List[str]:
        """Extract words from text for indexing with field-specific processing"""
        if not text:
            return []
        
        words = []
        
        # Extract compound tokens first
        compound_tokens = TextProcessor.extract_compound_tokens(text)
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
            and word.strip() not in TextProcessor.MTG_STOP_WORDS  # Filter stop words
        ]
        
        words.extend(single_words)
        return words

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for consistent matching"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def get_field_weight(field_name: str) -> float:
        """Get the importance weight for a specific field"""
        return TextProcessor.FIELD_WEIGHTS.get(field_name, 0.5)

    @staticmethod
    def is_stop_word(word: str) -> bool:
        """Check if a word is a MTG stop word"""
        return word.lower() in TextProcessor.MTG_STOP_WORDS
