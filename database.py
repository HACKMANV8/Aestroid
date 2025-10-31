from pymongo import MongoClient
from config import Config

class Database:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.MONGODB_DB_NAME]
        self.responses = self.db.responses
        
    def create_indexes(self):
        """Create database indexes for better query performance"""
        self.responses.create_index("timestamp")
        self.responses.create_index("response_id")
        
    def insert_response(self, response_data):
        """Insert a new chatbot response into the database"""
        return self.responses.insert_one(response_data)
    
    def update_response(self, response_id, update_data):
        """Update an existing response with new data"""
        return self.responses.update_one(
            {"response_id": response_id},
            {"$set": update_data}
        )
    
    def get_response(self, response_id):
        """Get a response by ID"""
        return self.responses.find_one({"response_id": response_id})
    
    def get_latest_response(self):
        """Get the most recent response"""
        return self.responses.find_one(sort=[("timestamp", -1)])
    
    def get_all_responses(self):
        """Get all responses"""
        return list(self.responses.find().sort("timestamp", -1))
    
    def update_with_location(self, response_id, location, timestamp):
        """Update response with location and timestamp from mobile app"""
        update_data = {
            "location": location,
            "mobile_timestamp": timestamp,
            "updated_at": timestamp
        }
        return self.update_response(response_id, update_data)

# Global database instance
db = Database()

