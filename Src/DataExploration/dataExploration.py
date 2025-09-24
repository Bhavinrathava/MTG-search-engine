import json
from pymongo import MongoClient
from collections import defaultdict
import re
import time
from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError, NetworkTimeout

def readSourceFile(filePath):
    with open(filePath, encoding="utf8") as f:
        return json.load(f)

def extract_words(text):
    """Extract words from text for indexing"""
    if not text:
        return []
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Split into words and filter out empty strings and very short words
    words = [word.strip() for word in text.split() if len(word.strip()) > 1]
    return words

def build_inverted_index_batch(cards_batch):
    """Build inverted index from a batch of cards"""
    inverted_index = defaultdict(set)
    
    for card in cards_batch:
        # Get card ID 
        card_id = card.get('uuid') 
        text_fields = []
        
        if 'originalText' in card:
            text_fields = extract_words(card['originalText'])
        
        # Add words to inverted index
        for word in text_fields:
            inverted_index[word].add(card_id)
    
    return inverted_index

def merge_inverted_indexes(main_index, batch_index):
    """Merge a batch inverted index into the main index"""
    for word, doc_ids in batch_index.items():
        if word in main_index:
            main_index[word].update(doc_ids)
        else:
            main_index[word] = doc_ids.copy()

def insert_cards_batch(collection, cards_batch, batch_num, total_batches):
    """Insert a batch of cards with retry logic"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            collection.insert_many(cards_batch, ordered=False)
            print(f"✓ Inserted card batch {batch_num}/{total_batches} ({len(cards_batch)} cards)")
            return True
        except (ServerSelectionTimeoutError, NetworkTimeout) as e:
            retry_count += 1
            print(f"⚠ Timeout on batch {batch_num}, retry {retry_count}/{max_retries}")
            if retry_count < max_retries:
                time.sleep(5)  # Wait before retry
            else:
                print(f"✗ Failed to insert batch {batch_num} after {max_retries} retries: {e}")
                return False
        except BulkWriteError as e:
            # Handle duplicate key errors or other bulk write issues
            print(f"⚠ Bulk write error on batch {batch_num}: {len(e.details.get('writeErrors', []))} errors")
            return True  # Continue even with some errors
        except Exception as e:
            print(f"✗ Unexpected error on batch {batch_num}: {e}")
            return False

def insert_index_batch(collection, index_batch, batch_num, total_batches):
    """Insert a batch of index documents with retry logic"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            collection.insert_many(index_batch, ordered=False)
            print(f"✓ Inserted index batch {batch_num}/{total_batches} ({len(index_batch)} words)")
            return True
        except (ServerSelectionTimeoutError, NetworkTimeout) as e:
            retry_count += 1
            print(f"⚠ Timeout on index batch {batch_num}, retry {retry_count}/{max_retries}")
            if retry_count < max_retries:
                time.sleep(5)  # Wait before retry
            else:
                print(f"✗ Failed to insert index batch {batch_num} after {max_retries} retries: {e}")
                return False
        except BulkWriteError as e:
            # Handle duplicate key errors
            print(f"⚠ Bulk write error on index batch {batch_num}: {len(e.details.get('writeErrors', []))} errors")
            return True  # Continue even with some errors
        except Exception as e:
            print(f"✗ Unexpected error on index batch {batch_num}: {e}")
            return False

def sendToMongoDB(cards, card_batch_size=500, index_batch_size=1000):
    # Connection with increased timeouts
    uri = "mongodb+srv://bhavinmongocluster.5t6smyb.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster"
    
    client = MongoClient(
        uri,
        tls=True,
        tlsCertificateKeyFile="Config/X509-cert-3753233507821277243.pem",
        serverSelectionTimeoutMS=60000,  # 60 seconds
        socketTimeoutMS=60000,           # 60 seconds  
        connectTimeoutMS=60000,          # 60 seconds
        maxPoolSize=10,                  # Limit connection pool size
        retryWrites=True
    )
    
    try:
        db = client['MTGCards']
        cards_collection = db['Cards']
        index_collection = db['InvertedIndex']
        
        # Test connection
        print("Testing database connection...")
        db.command('ping')
        print("✓ Database connection successful!")
        
        # Clear existing collections (optional - remove if you want to append)
        print("Clearing existing collections...")
        cards_collection.delete_many({})
        index_collection.delete_many({})
        print("✓ Collections cleared!")
        
        print(f"Starting insertion of {len(cards)} cards in batches of {card_batch_size}...")
        
        # Insert cards in batches
        total_card_batches = ((len(cards)//2) + card_batch_size - 1) // card_batch_size
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, (len(cards)//2), card_batch_size):
            batch = cards[i:i + card_batch_size]
            batch_num = (i // card_batch_size) + 1
            
            if insert_cards_batch(cards_collection, batch, batch_num, total_card_batches):
                successful_batches += 1
            else:
                failed_batches += 1
            
            # Small delay between batches to prevent overwhelming the server
            time.sleep(0.1)
        
        print(f"Cards insertion completed: {successful_batches} successful, {failed_batches} failed batches")
        
        # Build inverted index in batches
        print("Building inverted index in batches...")
        main_inverted_index = defaultdict(set)
        
        for i in range(0, len(cards), card_batch_size):
            batch = cards[i:i + card_batch_size]
            batch_num = (i // card_batch_size) + 1
            
            print(f"Processing index for batch {batch_num}/{total_card_batches}...")
            batch_index = build_inverted_index_batch(batch)
            merge_inverted_indexes(main_inverted_index, batch_index)
        
        print(f"✓ Inverted index built with {len(main_inverted_index)} unique words!")
        
        # Convert sets to lists and prepare for insertion
        print("Preparing index documents...")
        index_docs = []
        for word, doc_ids in main_inverted_index.items():
            index_docs.append({
                '_id': word,
                'documents': list(doc_ids),
                'count': len(doc_ids)
            })
        
        # Insert inverted index in batches
        if index_docs:
            print(f"Inserting inverted index in batches of {index_batch_size}...")
            total_index_batches = (len(index_docs) + index_batch_size - 1) // index_batch_size
            successful_index_batches = 0
            failed_index_batches = 0
            
            for i in range(0, len(index_docs), index_batch_size):
                batch = index_docs[i:i + index_batch_size]
                batch_num = (i // index_batch_size) + 1
                
                if insert_index_batch(index_collection, batch, batch_num, total_index_batches):
                    successful_index_batches += 1
                else:
                    failed_index_batches += 1
                
                # Small delay between batches
                time.sleep(0.1)
            
            print(f"Index insertion completed: {successful_index_batches} successful, {failed_index_batches} failed batches")
        
        # Create database indexes for better query performance
        print("Creating database indexes...")
        try:
            cards_collection.create_index("uuid")
            cards_collection.create_index("name")
            index_collection.create_index("count")
            print("✓ Database indexes created!")
        except Exception as e:
            print(f"⚠ Warning: Could not create indexes: {e}")
        
        print("🎉 All operations completed successfully!")
        
    except ServerSelectionTimeoutError as e:
        print(f"✗ Could not connect to MongoDB: {e}")
        print("Check your internet connection and MongoDB Atlas settings")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        client.close()

def query_inverted_index(search_word):
    """Helper function to query the inverted index"""
    uri = "mongodb+srv://bhavinmongocluster.5t6smyb.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster"
    
    client = MongoClient(
        uri,
        tls=True,
        tlsCertificateKeyFile="Config/X509-cert-3753233507821277243.pem",
        serverSelectionTimeoutMS=30000,
        socketTimeoutMS=30000,
        connectTimeoutMS=30000
    )
    
    try:
        db = client['MTGCards']
        index_collection = db['InvertedIndex']
        cards_collection = db['Cards']
        
        # Find word in inverted index
        result = index_collection.find_one({'_id': search_word.lower()})
        
        if result:
            print(f"Found '{search_word}' in {result['count']} documents")
            
            # Get the actual cards
            card_ids = result['documents']
            cards = list(cards_collection.find({'uuid': {'$in': card_ids}}))
            
            return cards
        else:
            print(f"Word '{search_word}' not found in index")
            return []
            
    finally:
        client.close()

if __name__ == "__main__":
    print("🃏 MTG Card Indexer Starting...")
    
    try:
        data = readSourceFile("Data/AllPrintings.json")
        cards = []
        
        for key in data['data'].keys():
            set_data = data['data'][key]
            cards.extend(set_data['cards'])
        
        print(f"✓ Loaded {len(cards)} cards from JSON")
        
        # You can adjust batch sizes based on your system and network
        # Smaller batches = more reliable but slower
        # Larger batches = faster but more prone to timeouts
        sendToMongoDB(cards, card_batch_size=500, index_batch_size=1000)
        
        # Example query
        print("\n🔍 Testing search functionality...")
        search_results = query_inverted_index("lightning")
        print(f"Found {len(search_results)} cards containing 'lightning'")
        
    except FileNotFoundError:
        print("✗ Could not find Data/AllPrintings.json file")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("Program finished.")