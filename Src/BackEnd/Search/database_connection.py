from pymongo import MongoClient
from typing import Optional
from pymongo.database import Database

class DatabaseConnection:
    """Manages MongoDB connection and provides shared connection pool"""
    
    _instance = None
    _mongo_client = None
    _mongo_db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.uri = "mongodb+srv://X509:@bhavinmongocluster.5t6smyb.mongodb.net/?authSource=$external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=BhavinMongoCluster"
        self.config = {
            'tls': True,
            'tlsCertificateKeyFile': os.path.join(current_dir, "Config", "X509-cert-3753233507821277243.pem"),
            'serverSelectionTimeoutMS': 30000,
            'socketTimeoutMS': 30000,
            'connectTimeoutMS': 30000,
            'maxPoolSize': 10,
            'minPoolSize': 2,
            'maxIdleTimeMS': 30000,
        }
    
    def get_connection(self) -> Optional[Database]:
        """Get or create a shared MongoDB connection"""
        if self._mongo_client is None:
            try:
                self._mongo_client = MongoClient(
                    self.uri,
                    **self.config
                )
                # Test the connection
                self._mongo_client.admin.command('ping')
                self._mongo_db = self._mongo_client['MTGCards']
            except Exception as e:
                print(f"Failed to connect to MongoDB: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                self._mongo_client = None
                self._mongo_db = None
                return None
                
        return self._mongo_db
    
    def close_connection(self):
        """Close the MongoDB connection if it exists"""
        if self._mongo_client:
            try:
                self._mongo_client.close()
            finally:
                self._mongo_client = None
                self._mongo_db = None
    
    def is_connected(self) -> bool:
        """Check if the database connection is active"""
        if self._mongo_client is None or self._mongo_db is None:
            return False
            
        try:
            # Test the connection
            self._mongo_client.admin.command('ping')
            return True
        except Exception:
            self._mongo_client = None
            self._mongo_db = None
            return False
