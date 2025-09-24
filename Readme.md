# Search Engine for Magic: the Gathering Cards 

This engine enables Natural Language Query over 110,000 cards from the universe of Magic : the Gathering and allows you to search for cards based on description. The Project focuses on fast search algorithms and using Semantic Similarity Based approach to recommend cards based on search Query. 


# How is the Project Structured 

The project consists of FrontEnd and a Backend Service Engine to separate the functionalities. The Project can be setup locally using ```Docker Compose up ```.
- We will be using Streamlit for Frontend for simplicity. 
- Backend will consist of the actual Search Module and an API exposing search Functionality through a ```/search POST``` endpoint.
- The API will be written in ```Flask```


# Search Algorithm Deep Dive

The search system uses a sophisticated multi-layered approach to provide both fast exact matches and intelligent fuzzy searching:

## 1. Fast Name Search (TRIE-Based)
For queries that likely reference specific card names (< 6 words), the system first attempts a fast TRIE-based search:

a) **Exact Match**:
- Uses an in-memory TRIE data structure mapping card names to UUIDs
- Normalizes text (lowercase, remove special chars)
- Returns immediately if exact match found

b) **Prefix Match**:
- If no exact match, tries matching query as a prefix
- Returns cards starting with the query text
- Limited to top 5 matches for performance

c) **Fuzzy Match** (fallback):
The fuzzy matching algorithm is a sophisticated two-phase process:

1. **Text Normalization**:
   ```python
   # Example: "Lightening Bolt!" -> "lightening bolt"
   normalized_text = text.lower()
   normalized_text = remove_special_chars(normalized_text)
   normalized_text = standardize_spaces(normalized_text)
   ```

2. **Word-Based Phase**:
   - Splits query into words
   - Searches for cards containing ANY of these words
   - Example:
     ```python
     Query: "lighting bolt"
     Words: ["lighting", "bolt"]
     Matches: 
     - "Lightning Bolt" (contains "bolt")
     - "Lightning Strike" (contains "lightning")
     - "Chain Lightning" (contains "lightning")
     ```

3. **Similarity Calculation**:
   - For word-based matches, calculates similarity using SequenceMatcher
   - Example similarity scores:
     ```python
     "lighting bolt" vs "lightning bolt" = 0.91 (high match)
     "lighting bolt" vs "lightning strike" = 0.73 (partial match)
     "lighting bolt" vs "shock" = 0.12 (no match)
     ```

4. **Filtering**:
   - Word-based matches must have similarity > 0.7
   - Other potential matches must have:
     - Length difference ≤ 2 characters
     - Similarity > 0.6
   - Results are sorted by similarity score

5. **Caching**:
   ```python
   cache_key = f"fuzzy_{normalized_query}"
   cache[cache_key] = results  # Cache for future use
   ```

## 2. Full Search Pipeline
For longer/descriptive queries or if fast search yields no results:

### a) Query Expansion
- Removes MTG-specific stop words (e.g., "card", "creature", "mana")
- Expands query using:
  - MTG-specific synonyms dictionary
  - WordNet general synonyms
- Handles compound tokens (e.g., "Serra Angel", "X of Y" patterns)
- Generates query embeddings using MiniLM-V2

### b) Inverted Index Search
The inverted index search is a critical component that efficiently finds relevant cards:

1. **Index Structure**:
   ```python
   {
     "word": {
       "doc_frequency": float,  # How common the word is
       "importance_score": float,  # Word significance
       "field_counts": {
         "name": int,      # Occurrences in names
         "type": int,      # Occurrences in types
         "text": int,      # Occurrences in rules text
         # etc.
       },
       "documents": [card_ids]  # Cards containing this word
     }
   }
   ```

2. **Search Process**:
   ```python
   # Example query: "powerful red dragon"
   
   # Step 1: Extract words
   words = ["powerful", "red", "dragon"]
   
   # Step 2: Score each word
   word_scores = {
     "powerful": 0.8,  # Less common = higher score
     "red": 0.5,      # Common color = lower score
     "dragon": 0.9    # Specific type = higher score
   }
   
   # Step 3: Calculate card scores
   card_scores = {
     "card_id": word_score * field_weight
   }
   ```

3. **Field Weighting**:
   - Name matches (1.0): Most important
   - Type matches (0.8): High importance
   - Subtype matches (0.7): Good relevance
   - Text matches (0.6): Relevant but not primary
   - Flavor matches (0.3): Least important

4. **Filtering**:
   - Removes very common words (doc_frequency > 0.1)
   - Example: "card", "the", "a" are ignored

5. **Output Generation**:
   ```python
   # Returns sorted list of card IDs
   return [
     card_id for card_id, score 
     in sorted(card_scores.items(), key=lambda x: x[1], reverse=True)
   ]
   ```

### c) MongoDB Query Parameters
- Uses GPT to generate MongoDB query parameters
- Converts natural language to structured filters
- Falls back to inverted index results if no matches

### d) BM25 Scoring
- Implements Okapi BM25 ranking
- Weights different card fields
- Combines text from multiple fields
- Provides relevance-based ranking

### e) Semantic Search
- Uses MiniLM-V2 embeddings
- Calculates cosine similarity between:
  - Expanded query embedding
  - Card text embeddings
- Re-ranks results based on semantic similarity

### f) Post-Processing
- Deduplicates cards (removes duplicate printings)
- Caches results for performance
- Returns final ranked list

## Performance Optimizations

1. **Caching**:
- Embedding model cached globally
- Fuzzy search results cached (up to 1000 queries)
- Database connection pooling (2-10 connections)

2. **Text Processing**:
- MTG-specific stop words
- Field-specific weights
- Compound token handling
- Length-based filtering

3. **Database**:
- Connection pooling
- Timeout handling
- Persistent connections
- Inverted indexing

4. **Search Flow**:
- Progressive filtering
- Early exit on exact matches
- Parallel processing where possible
- Fallback mechanisms

This multi-layered approach allows the system to handle both quick card name lookups and complex descriptive queries efficiently, while maintaining high accuracy and relevance in search results.
