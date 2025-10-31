import os
import json
import uuid
from groq import Groq
from config import Config

class GroqChatbotService:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL
        self.jsonl_file = Config.JSONL_CONTEXT_FILE
        self.conversation_history = []
        
    def load_jsonl_context(self):
        """Load context from JSONL file"""
        context = []
        if os.path.exists(self.jsonl_file):
            try:
                with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            context.append(json.loads(line))
            except Exception as e:
                print(f"Error loading JSONL context: {e}")
        return context
    
    def get_chat_history_from_db(self, response_id=None):
        """Get conversation history from database if response_id is provided"""
        if response_id:
            # Lazy import to avoid circular dependencies
            from database import db
            previous_responses = db.get_all_responses()
            # Filter to get responses before the current one if needed
            history = []
            for resp in previous_responses:
                if resp.get('messages'):
                    messages = resp['messages']
                    # Only add message dicts with role and content
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                history.append(msg)
            return history
        return []
    
    def build_messages(self, user_input, response_id=None):
        """Build messages array with context from JSONL and previous conversations"""
        messages = []
        
        # Load JSONL context
        jsonl_context = self.load_jsonl_context()
        if jsonl_context:
            # Add JSONL context as system message or initial context
            context_text = json.dumps(jsonl_context, indent=2)
            messages.append({
                "role": "system",
                "content": f"Additional context from JSONL file:\n{context_text}\n\nUse this context when responding to user queries."
            })
        
        # Get previous conversation history from database
        db_history = self.get_chat_history_from_db(response_id)
        
        # Add conversation history (last 10 messages for context management)
        for msg in db_history[-10:]:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                messages.append(msg)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def get_chatbot_response(self, user_input, response_id=None):
        """Get response from Groq chatbot with context"""
        try:
            messages = self.build_messages(user_input, response_id)
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=2048
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Store this interaction in conversation history
            conversation_data = {
                "user_message": user_input,
                "assistant_message": response_text,
                "messages": messages + [{
                    "role": "assistant",
                    "content": response_text
                }]
            }
            
            return response_text, conversation_data
            
        except Exception as e:
            print(f"Error getting chatbot response: {e}")
            return f"Error: {str(e)}", None
    
    def process_update(self, update_data, response_id=None):
        """Process updates from mobile app or pygame and get updated strategy"""
        # Convert update to user input format
        user_input = json.dumps(update_data) if isinstance(update_data, dict) else str(update_data)
        user_input = f"Update received: {user_input}. Please provide updated strategy based on this information."
        
        return self.get_chatbot_response(user_input, response_id)

