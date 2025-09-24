from collections import OrderedDict
from threading import Lock
import json

class SearchCache:
    """Thread-safe LRU cache for search results with size limit"""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()  # {query: results}
        self.lock = Lock()  # For thread safety
        
    def get(self, query: str) -> dict:
        """Get cached results for a query"""
        with self.lock:
            if query in self.cache:
                # Move to end (most recently used)
                results = self.cache.pop(query)
                self.cache[query] = results
                return results
            return None
            
    def put(self, query: str, results: dict):
        """Cache results for a query"""
        with self.lock:
            if query in self.cache:
                # Update existing entry
                self.cache.pop(query)
            elif len(self.cache) >= self.max_size:
                # Remove oldest entry if cache is full
                self.cache.popitem(last=False)
            
            # Add new entry
            self.cache[query] = results
            
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            
    def get_size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
