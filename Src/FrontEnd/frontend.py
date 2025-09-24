import streamlit as st
import requests
import time
import json
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="MTG Card Search",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .search-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0;
    }
    
    .search-input {
        flex: 1;
    }
    
    .search-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .search-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        font-size: 18px;
        color: #667eea;
    }
    
    .card-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .card-name {
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-bottom: 8px;
    }
    
    .card-type {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .card-text {
        font-size: 16px;
        color: #444;
        line-height: 1.4;
        margin: 10px 0;
    }
    
    .card-stats {
        display: flex;
        gap: 15px;
        font-size: 14px;
        color: #888;
        margin-top: 10px;
    }
    
    .no-results {
        text-align: center;
        padding: 40px;
        color: #666;
        font-size: 18px;
    }
    
    .search-stats {
        color: #666;
        font-size: 14px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def search_api(query: str) -> Dict[str, Any]:
    """
    Make API call to search endpoint
    """
    import os
    api_url = f"{os.getenv('BACKEND_URL', 'http://localhost:5000')}/search"
    
    try:
        # Simulate API call
        response = requests.post(
            api_url,
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code: {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        # For demo purposes, return mock data when API is not available
        time.sleep(2)  # Simulate API delay
        return get_mock_search_results(query)
    except requests.exceptions.Timeout:
        return {"error": "Search request timed out"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

def get_mock_search_results(query: str) -> Dict[str, Any]:
    """
    Mock search results for demonstration
    Remove this function when you have a real API
    """
    mock_cards = [
        {
            "id": "1",
            "name": "Lightning Bolt",
            "type": "Instant",
            "manaCost": "{R}",
            "text": "Lightning Bolt deals 3 damage to any target.",
            "set": "Alpha",
            "rarity": "Common",
            "power": None,
            "toughness": None
        },
        {
            "id": "2", 
            "name": "Lightning Strike",
            "type": "Instant",
            "manaCost": "{1}{R}",
            "text": "Lightning Strike deals 3 damage to any target.",
            "set": "Magic 2014",
            "rarity": "Common",
            "power": None,
            "toughness": None
        },
        {
            "id": "3",
            "name": "Lightning Elemental",
            "type": "Creature — Elemental",
            "manaCost": "{3}{R}",
            "text": "Haste (This creature can attack and {T} as soon as it comes under your control.)",
            "set": "Portal",
            "rarity": "Uncommon",
            "power": "4",
            "toughness": "1"
        }
    ]
    
    # Filter mock results based on query
    filtered_cards = [
        card for card in mock_cards 
        if query.lower() in card["name"].lower() or query.lower() in card["text"].lower()
    ]
    
    return {
        "cards": filtered_cards,
        "count": len(filtered_cards),
        "status": "success"
    }

def display_card(card: Dict[str, Any]):
    """Display a single card in a styled container"""
    with st.container():
        st.markdown(f"""
        <div class="card-container">
            <div class="card-name">{card.get('name', 'Unknown Card')}</div>
            <div class="card-type">{card.get('type', 'Unknown Type')} • {card.get('set', 'Unknown Set')} • {card.get('rarity', 'Unknown Rarity')}</div>
            <div class="card-text">{card.get('text', 'No card text available.')}</div>
            <div class="card-stats">
                <span>Mana Cost: {card.get('manaCost', 'N/A')}</span>
                {f"<span>Power/Toughness: {card.get('power', '')}/{card.get('toughness', '')}</span>" if card.get('power') and card.get('toughness') else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.title("🃏 MTG Card Search")
    st.markdown("Search through Magic: The Gathering cards using our inverted index!")
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'is_searching' not in st.session_state:
        st.session_state.is_searching = False
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # Search interface
    col1, col2 = st.columns([6, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for cards...",
            placeholder="Enter card name, text, or keywords (e.g., 'lightning', 'creature', 'haste')",
            key="search_input",
            label_visibility="collapsed"
        )
    
    with col2:
        search_clicked = st.button(
            "🔍", 
            key="search_button",
            help="Search cards",
            use_container_width=True
        )
    
    # Handle search
    if search_clicked and search_query.strip():
        st.session_state.is_searching = True
        st.session_state.last_query = search_query.strip()
        st.rerun()
    
    # Handle Enter key press
    if search_query != st.session_state.last_query and search_query.strip():
        if st.session_state.get('enter_pressed', False):
            st.session_state.is_searching = True
            st.session_state.last_query = search_query.strip()
            st.rerun()
    
    # Perform search if needed
    if st.session_state.is_searching and st.session_state.last_query:
        with st.spinner('🔄 Searching cards...'):
            search_results = search_api(st.session_state.last_query)
            st.session_state.search_results = search_results
            st.session_state.is_searching = False
        st.rerun()
    
    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        if "error" in results:
            st.error(f"❌ Search Error: {results['error']}")
            st.info("💡 Make sure your API endpoint is running and accessible")
            
        elif "cards" in results:
            cards = results["cards"]
            total_count = results.get("count", len(cards))
            query_time = 0  # Backend doesn't provide query time
            
            # Search statistics
            st.markdown(f"""
            <div class="search-stats">
                Found {total_count} card(s) for "{st.session_state.last_query}" in {query_time:.3f} seconds
            </div>
            """, unsafe_allow_html=True)
            
            if cards:
                # Display cards
                for card in cards:
                    display_card(card)
                    
                # Pagination info (if applicable)
                if len(cards) < total_count:
                    st.info(f"Showing first {len(cards)} results out of {total_count}")
                    
            else:
                st.markdown("""
                <div class="no-results">
                    🔍 No cards found matching your search.<br>
                    Try different keywords or check your spelling.
                </div>
                """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        **Search Tips:**
        - Search by card name: `"Lightning Bolt"`
        - Search by card type: `"instant"`, `"creature"`
        - Search by keywords: `"haste"`, `"flying"`, `"trample"`
        - Search by mana cost: `"red"`, `"blue"`
        - Use multiple words: `"red creature haste"`
        
        **Features:**
        - Real-time search with loading indicator
        - Detailed card information display
        - Search statistics and result count
        - Error handling for API issues
        
        **Note:** This demo uses mock data when the API is not available.
        Replace the API endpoint in the code with your actual search service.
        """)
    
    # API endpoint info
    with st.expander("🔧 API Configuration", expanded=False):
        import os
        backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')
        st.code(
            f"# Current API endpoint\n"
            f"POST {backend_url}/search\n\n"
            "# Request format:\n"
            "{\n"
            '    "query": "your search terms"\n'
            "}\n\n"
            "# Response format:\n"
            "{\n"
            '    "cards": [...],\n'
            '    "count": 123,\n'
            '    "status": "success"\n'
            "}", 
            language="json")
        
        st.info("💡 Update the `api_url` variable in the code to point to your actual search API endpoint.")

if __name__ == "__main__":
    main()
