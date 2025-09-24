from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardEmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.client = None
        self.db = None
        self.db_connected = False
        self.model_loaded = False
        
        # MongoDB connection details
        self.connection_config = {
            'uri': "mongodb+srv://bhavinmongocluster.5t6smyb.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster",
            'tls': True,
            'tlsCertificateKeyFile': "Config/X509-cert-3753233507821277243.pem",
            'serverSelectionTimeoutMS': 60000,
            'socketTimeoutMS': 60000,
            'connectTimeoutMS': 60000
        }
        
    def setup_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model_loaded = True
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_database(self):
        """Setup MongoDB connection"""
        try:
            logger.info("Connecting to MongoDB...")
            self.client = MongoClient(
                self.connection_config['uri'],
                tls=self.connection_config['tls'],
                tlsCertificateKeyFile=self.connection_config['tlsCertificateKeyFile'],
                serverSelectionTimeoutMS=self.connection_config['serverSelectionTimeoutMS'],
                socketTimeoutMS=self.connection_config['socketTimeoutMS'],
                connectTimeoutMS=self.connection_config['connectTimeoutMS']
            )
            
            self.db = self.client['MTGCards']
            
            # Test connection
            self.db.command('ping')
            self.db_connected = True
            logger.info("✓ Database connection successful!")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.db_connected = False
            raise
    
    def generate_embedding(self, text: str) -> Optional[list]:
        """
        Generate embedding for a text string
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of float values representing the embedding, or None if failed
        """
        if not text or not text.strip():
            return None
            
        try:
            # Generate embedding
            embedding = self.model.encode(text.strip())
            
            # Convert numpy array to list for MongoDB storage
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None
    
    def process_cards_batch(self, batch_size: int = 100):
        """
        Process cards in batches to add embeddings
        
        Args:
            batch_size: Number of cards to process in each batch
        """
        if not self.model_loaded or not self.db_connected:
            raise ValueError("Model and database must be setup before processing")
        
        cards_collection = self.db['Cards']
        
        # Get total count of cards
        total_cards = cards_collection.count_documents({})
        logger.info(f"Total cards in collection: {total_cards}")
        
        # Count cards that already have embeddings
        cards_with_embeddings = cards_collection.count_documents({"embedding": {"$exists": True}})
        logger.info(f"Cards already with embeddings: {cards_with_embeddings}")
        
        # Find cards without embeddings
        query = {"embedding": {"$exists": False}}
        remaining_cards = cards_collection.count_documents(query)
        logger.info(f"Cards to process: {remaining_cards}")
        
        if remaining_cards == 0:
            logger.info("All cards already have embeddings!")
            return
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        # Process in batches
        cursor = cards_collection.find(query).batch_size(batch_size)
        
        batch = []
        
        for card in cursor:
            batch.append(card)
            
            # Process batch when it's full
            if len(batch) >= batch_size:
                batch_processed, batch_failed = self._process_batch(batch, cards_collection)
                processed += batch_processed
                failed += batch_failed
                
                # Log progress
                elapsed_time = time.time() - start_time
                cards_per_second = processed / elapsed_time if elapsed_time > 0 else 0
                
                logger.info(f"Progress: {processed}/{remaining_cards} cards processed "
                          f"({processed/remaining_cards*100:.1f}%) - "
                          f"{cards_per_second:.1f} cards/sec - "
                          f"{failed} failed")
                
                batch = []
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
        
        # Process remaining cards in the last batch
        if batch:
            batch_processed, batch_failed = self._process_batch(batch, cards_collection)
            processed += batch_processed
            failed += batch_failed
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"✓ Processing complete!")
        logger.info(f"  Total processed: {processed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        if total_time > 0:
            logger.info(f"  Average speed: {processed/total_time:.2f} cards/second")
    
    def _process_batch(self, batch, collection):
        """Process a batch of cards"""
        processed = 0
        failed = 0
        
        for card in batch:
            try:
                card_id = card.get('uuid') or card.get('_id')
                original_text = card.get('originalText', '')
                
                # Skip if no original text
                if not original_text or not original_text.strip():
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(original_text)
                
                if embedding is not None:
                    # Update the card document with the embedding
                    result = collection.update_one(
                        {'_id': card['_id']},
                        {
                            '$set': {
                                'embedding': embedding,
                                'embedding_model': self.model_name,
                                'embedding_created_at': time.time()
                            }
                        }
                    )
                    
                    if result.modified_count > 0:
                        processed += 1
                    else:
                        logger.warning(f"Failed to update card: {card_id}")
                        failed += 1
                else:
                    logger.warning(f"Failed to generate embedding for card: {card_id}")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing card {card.get('uuid', 'unknown')}: {e}")
                failed += 1
        
        return processed, failed
    
    def create_embedding_index(self):
        """Create an index on the embedding field for better performance"""
        if not self.db_connected:
            logger.error("No database connection available")
            return
            
        try:
            cards_collection = self.db['Cards']
            
            # Create index on embedding field
            cards_collection.create_index("embedding")
            logger.info("✓ Created index on embedding field")
            
            # Create compound index for embedding searches
            cards_collection.create_index([("embedding", 1), ("name", 1)])
            logger.info("✓ Created compound index on embedding and name fields")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def verify_embeddings(self, sample_size: int = 10):
        """Verify that embeddings were created correctly"""
        if not self.db_connected:
            logger.error("No database connection available")
            return
            
        try:
            cards_collection = self.db['Cards']
            
            # Get sample of cards with embeddings
            sample_cards = list(cards_collection.find(
                {"embedding": {"$exists": True}},
                {"name": 1, "originalText": 1, "embedding": 1, "embedding_model": 1}
            ).limit(sample_size))
            
            logger.info(f"Verification - Sample of {len(sample_cards)} cards with embeddings:")
            
            for card in sample_cards:
                embedding_length = len(card.get('embedding', []))
                text_preview = card.get('originalText', '')[:50] + "..." if len(card.get('originalText', '')) > 50 else card.get('originalText', '')
                
                logger.info(f"  - {card.get('name', 'Unknown')}: "
                          f"embedding_length={embedding_length}, "
                          f"model={card.get('embedding_model', 'unknown')}, "
                          f"text='{text_preview}'")
            
            # Get statistics
            total_cards = cards_collection.count_documents({})
            cards_with_embeddings = cards_collection.count_documents({"embedding": {"$exists": True}})
            
            logger.info(f"Embedding Statistics:")
            logger.info(f"  Total cards: {total_cards}")
            logger.info(f"  Cards with embeddings: {cards_with_embeddings}")
            if total_cards > 0:
                logger.info(f"  Completion rate: {cards_with_embeddings/total_cards*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
    
    def close_connection(self):
        """Close the database connection"""
        if self.client is not None:
            self.client.close()
            self.db_connected = False
            logger.info("Database connection closed")

def main():
    """Main function to run the embedding generation"""
    generator = CardEmbeddingGenerator()
    
    try:
        # Setup
        generator.setup_model()
        generator.setup_database()
        
        # Process cards
        logger.info("Starting embedding generation...")
        generator.process_cards_batch(batch_size=5000)  # Adjust batch size as needed
        
        # Create indexes
        generator.create_embedding_index()
        
        # Verify results
        generator.verify_embeddings(sample_size=5)
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
    
    finally:
        generator.close_connection()
        logger.info("Script completed!")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import sentence_transformers
    except ImportError:
        print("Please install sentence-transformers: pip install sentence-transformers")
        exit(1)
    
    main()