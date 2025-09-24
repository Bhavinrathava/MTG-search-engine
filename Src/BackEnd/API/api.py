from flask import Flask, request, jsonify
from flask_cors import CORS
from BackEnd.Search.query_processing import QueryProcessing
from bson import ObjectId
from .cache import SearchCache
import json
import logging

logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle MongoDB ObjectId and other special types"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = JSONEncoder

# Initialize cache with max size of 1000 queries
search_cache = SearchCache(max_size=1000)

@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint that processes queries and returns matching cards
    
    Expected JSON body:
    {
        "query": "search query text",
        "k": 10  // optional, number of results to return
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query parameter',
                'status': 'error'
            }), 400
        
        query = data['query']
        k = data.get('k', 10)  # Default to 10 results if k not specified
        
        logger.info("Processing search request for query: %s", query)
        
        # Check cache first
        cache_key = f"{query}:{k}"
        cached_results = search_cache.get(cache_key)
        if cached_results is not None:
            logger.info("Cache hit for query: %s", query)
            return jsonify(cached_results)
        
        # Cache miss - Initialize query processor
        processor = QueryProcessing(query)
        
        # Get results
        results = processor.findTopK(k)
        
        # Filter only required fields for each card
        filtered_results = []
        for card in results:
            filtered_card = {
                'name': card.get('name'),
                'type': card.get('type'),
                'set': card.get('set'),
                'rarity': card.get('rarity'),
                'text': card.get('text'),
                'manaCost': card.get('manaCost'),
                'power': card.get('power'),
                'toughness': card.get('toughness')
            }
            filtered_results.append(filtered_card)
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(json.dumps(filtered_results, cls=JSONEncoder))
        
        logger.info("Search completed successfully with %d results", len(results))
        response = {
            'cards': serializable_results,
            'count': len(serializable_results),
            'status': 'success'
        }
        
        # Cache the results
        search_cache.put(cache_key, response)
        logger.info("Cached results for query: %s", query)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error("Error processing search request: %s", str(e), exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MTG Card Search API'
    })

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    return jsonify({
        'size': search_cache.get_size(),
        'max_size': search_cache.max_size
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the cache"""
    search_cache.clear()
    return jsonify({
        'status': 'success',
        'message': 'Cache cleared'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
