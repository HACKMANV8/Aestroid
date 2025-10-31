# Chatbot Backend with Flask, MongoDB, and Groq API

A comprehensive Flask backend that integrates with Groq API for chatbot functionality, MongoDB for data storage, and WebSocket support for real-time updates.

## Features

- **Chatbot Integration**: Get responses from Groq API chatbot with context management
- **MongoDB Storage**: Store all chatbot responses with metadata
- **Mobile App Integration**: Update responses with location and timestamp data
- **Pygame Integration**: Receive updates from pygame application and get updated strategies
- **Real-time Updates**: WebSocket support for real-time response updates to website
- **Context Management**: Maintains conversation history and loads context from JSONL file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Required environment variables:
- `MONGODB_URI`: MongoDB connection string (default: mongodb://localhost:27017/)
- `MONGODB_DB_NAME`: Database name (default: chatbot_db)
- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: Groq model to use (default: llama-3.1-70b-versatile)
- `SECRET_KEY`: Flask secret key
- `JSONL_CONTEXT_FILE`: Path to JSONL context file (default: context.jsonl)

### 3. Install and Start MongoDB

#### For Arch Linux:

MongoDB is not in the official repositories, but available in AUR:

```bash
# Install MongoDB from AUR (using yay)
yay -S mongodb-bin

# Or if you have paru instead:
# paru -S mongodb-bin

# Start MongoDB service
sudo systemctl start mongodb

# Enable MongoDB to start on boot (optional)
sudo systemctl enable mongodb

# Check if MongoDB is running
sudo systemctl status mongodb
```

**Note:** If you don't want to install MongoDB locally, use MongoDB Atlas (cloud) option below.

#### For other Linux distributions:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mongodb
sudo systemctl start mongod
sudo systemctl enable mongod

# Or use the official MongoDB installation guide:
# https://www.mongodb.com/docs/manual/installation/
```

#### Alternative: Use MongoDB Atlas (Cloud)

If you prefer not to install MongoDB locally, you can use MongoDB Atlas (free tier):

1. Sign up at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Get your connection string
4. Update `MONGODB_URI` in your `.env` file with the Atlas connection string

Example `.env`:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

### 4. Run the Application

**Using the run script (recommended):**
```bash
./run.sh
```

**Or manually:**
```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python app.py
```

The server will start on `http://0.0.0.0:5000`

**Note:** Make sure MongoDB is running before starting the application:
```bash
sudo systemctl start mongodb
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Chatbot Query
```
POST /api/chatbot/query
Body: {
    "query": "Your question here"
}
Response: {
    "response_id": "uuid",
    "response": "Chatbot response",
    "timestamp": "ISO timestamp"
}
```

### Mobile App Update
```
POST /api/mobile/update
Body: {
    "response_id": "uuid",
    "location": {
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Pygame Update
```
POST /api/pygame/update
Body: {
    "response_id": "uuid (optional)",
    "update_data": {
        "key": "value",
        "map_data": "..."
    }
}
```

### Get All Responses
```
GET /api/responses
```

### Get Specific Response
```
GET /api/responses/<response_id>
```

## WebSocket Events

### Client Events
- `connect`: Client connects to server
- `subscribe_updates`: Subscribe to response updates

### Server Events
- `connected`: Connection confirmed
- `subscribed`: Subscription confirmed
- `response_updated`: Emitted whenever a response is created or updated
  ```json
  {
      "response_id": "uuid",
      "response": "Response text",
      "timestamp": "ISO timestamp",
      "location": {...},
      "update_source": "pygame" (optional)
  }
  ```

## Website Integration (Example)

To receive real-time updates on your website:

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected to backend');
    socket.emit('subscribe_updates');
});

socket.on('response_updated', (data) => {
    console.log('New response:', data);
    // Update your UI with the new response
    updateResponseDisplay(data);
});
```

## Project Structure

```
backend/
├── app.py              # Main Flask application with SocketIO
├── config.py           # Configuration settings
├── database.py         # MongoDB database operations
├── groq_service.py     # Groq API chatbot service
├── routes.py           # API endpoints
├── context.jsonl       # Context file for chatbot
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Context Management

The chatbot maintains context through:
1. **JSONL File**: Static context loaded from `context.jsonl`
2. **Database History**: Previous conversation messages stored in MongoDB
3. **Current Session**: Messages from the current conversation

The chatbot uses the last 10 messages from database history plus JSONL context for each query.

## Notes

- Make sure to set your `GROQ_API_KEY` in the `.env` file
- MongoDB should be running before starting the application
- The application uses Flask-SocketIO with eventlet for WebSocket support
- All timestamps are stored in ISO format (UTC)

