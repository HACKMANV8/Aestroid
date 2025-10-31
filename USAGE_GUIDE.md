# Backend Usage Guide

This guide explains how to use all the features of the Flask chatbot backend.

## Table of Contents
1. [Starting the Server](#starting-the-server)
2. [Basic API Usage](#basic-api-usage)
3. [Mobile App Integration](#mobile-app-integration)
4. [Pygame Integration](#pygame-integration)
5. [Real-time Updates (WebSocket)](#real-time-updates-websocket)
6. [Complete Examples](#complete-examples)

---

## Starting the Server

### Option 1: Using the run script
```bash
cd /home/shankhanil/hackman_v8/backend
./run.sh
```

### Option 2: Manual start
```bash
cd /home/shankhanil/hackman_v8/backend
source venv/bin/activate
python app.py
```

The server will start on `http://localhost:5000` (or `http://0.0.0.0:5000`)

**Before starting:** Make sure MongoDB is running:
```bash
sudo systemctl start mongodb
```

---

## Basic API Usage

### 1. Chatbot Query

Get a response from the chatbot with context from previous conversations and JSONL file.

**Endpoint:** `POST /api/chatbot/query`

**Request:**
```bash
curl -X POST http://localhost:5000/api/chatbot/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What strategy should I use for this situation?"
  }'
```

**Response:**
```json
{
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "Based on the current context, I recommend...",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Python Example:**
```python
import requests

response = requests.post('http://localhost:5000/api/chatbot/query', 
    json={'query': 'What should I do next?'})
data = response.json()
print(f"Response ID: {data['response_id']}")
print(f"Response: {data['response']}")
```

**JavaScript Example:**
```javascript
fetch('http://localhost:5000/api/chatbot/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        query: 'What should I do next?'
    })
})
.then(response => response.json())
.then(data => {
    console.log('Response ID:', data.response_id);
    console.log('Response:', data.response);
});
```

### 2. Get All Responses

**Endpoint:** `GET /api/responses`

```bash
curl http://localhost:5000/api/responses
```

### 3. Get Specific Response

**Endpoint:** `GET /api/responses/<response_id>`

```bash
curl http://localhost:5000/api/responses/550e8400-e29b-41d4-a716-446655440000
```

### 4. Health Check

**Endpoint:** `GET /api/health`

```bash
curl http://localhost:5000/api/health
```

---

## Mobile App Integration

Update a chatbot response with location and timestamp data from your mobile app.

**Endpoint:** `POST /api/mobile/update`

**Request:**
```json
{
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 10.5
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Full Example (Android/Kotlin):**
```kotlin
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

fun updateResponseWithLocation(responseId: String, lat: Double, lon: Double) {
    val client = OkHttpClient()
    
    val json = JSONObject().apply {
        put("response_id", responseId)
        put("location", JSONObject().apply {
            put("latitude", lat)
            put("longitude", lon)
        })
        put("timestamp", java.time.Instant.now().toString())
    }
    
    val mediaType = "application/json".toMediaType()
    val body = json.toString().toRequestBody(mediaType)
    
    val request = Request.Builder()
        .url("http://your-server-ip:5000/api/mobile/update")
        .post(body)
        .build()
    
    client.newCall(request).execute().use { response ->
        if (response.isSuccessful) {
            println("Update successful!")
        }
    }
}
```

**Full Example (React Native/JavaScript):**
```javascript
const updateLocation = async (responseId, location) => {
    try {
        const response = await fetch('http://your-server-ip:5000/api/mobile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                response_id: responseId,
                location: {
                    latitude: location.latitude,
                    longitude: location.longitude,
                },
                timestamp: new Date().toISOString()
            })
        });
        
        const data = await response.json();
        console.log('Location updated:', data);
    } catch (error) {
        console.error('Error updating location:', error);
    }
};
```

**Python Example (Mobile App Backend):**
```python
import requests
from datetime import datetime

def update_mobile_location(response_id, latitude, longitude):
    url = "http://localhost:5000/api/mobile/update"
    data = {
        "response_id": response_id,
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    response = requests.post(url, json=data)
    return response.json()
```

---

## Pygame Integration

Send map updates from your pygame application and get updated chatbot strategies.

**Endpoint:** `POST /api/pygame/update`

**Request (New Response):**
```json
{
  "update_data": {
    "map_state": "initial",
    "player_position": {"x": 0, "y": 0},
    "enemies": [],
    "events": []
  }
}
```

**Request (Update Existing Response):**
```json
{
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "update_data": {
    "map_state": "updated",
    "player_position": {"x": 100, "y": 200},
    "enemies": [{"id": 1, "x": 150, "y": 250}],
    "events": ["enemy_spawned", "item_collected"]
  }
}
```

**Python Example (Pygame):**
```python
import requests
import json

def send_pygame_update(map_data, response_id=None):
    """
    Send map update to chatbot backend
    
    Args:
        map_data: Dictionary containing map state, player position, etc.
        response_id: Optional response ID to continue existing conversation
    """
    url = "http://localhost:5000/api/pygame/update"
    
    payload = {
        "update_data": map_data
    }
    
    if response_id:
        payload["response_id"] = response_id
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        
        print(f"New Strategy: {data['response']}")
        print(f"Response ID: {data['response_id']}")
        
        return data
    except Exception as e:
        print(f"Error sending update: {e}")
        return None

# Example usage in pygame game loop
def game_update():
    # Your game logic here
    map_state = {
        "map_state": "updated",
        "player_position": {"x": player.x, "y": player.y},
        "enemies": [{"id": e.id, "x": e.x, "y": e.y} for e in enemies],
        "events": current_events
    }
    
    # Send update to chatbot
    result = send_pygame_update(map_state, current_response_id)
    
    if result:
        current_response_id = result['response_id']
        # Use the updated strategy from result['response']
```

**Complete Pygame Integration Example:**
```python
import pygame
import requests
import threading

class ChatbotIntegration:
    def __init__(self):
        self.api_url = "http://localhost:5000/api/pygame/update"
        self.response_id = None
        self.current_strategy = None
    
    def update_map(self, game_state):
        """Send map update and get new strategy"""
        payload = {
            "update_data": {
                "map_state": game_state.get('map_state', 'unknown'),
                "player_position": game_state.get('player_pos', {}),
                "enemies": game_state.get('enemies', []),
                "items": game_state.get('items', []),
                "events": game_state.get('events', [])
            }
        }
        
        if self.response_id:
            payload["response_id"] = self.response_id
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=5)
            data = response.json()
            
            self.response_id = data['response_id']
            self.current_strategy = data['response']
            
            return self.current_strategy
        except Exception as e:
            print(f"Error updating chatbot: {e}")
            return None

# In your pygame main loop
chatbot = ChatbotIntegration()

while running:
    # Game logic...
    
    # Periodically update chatbot with map state
    if frame_count % 60 == 0:  # Update every 60 frames
        game_state = {
            'map_state': 'active',
            'player_pos': {'x': player.rect.x, 'y': player.rect.y},
            'enemies': [{'x': e.rect.x, 'y': e.rect.y} for e in enemies],
            'events': []
        }
        
        strategy = chatbot.update_map(game_state)
        if strategy:
            print(f"Strategy: {strategy}")
```

---

## Real-time Updates (WebSocket)

The backend automatically emits WebSocket events whenever a response is created or updated. This allows your website to receive real-time updates without polling.

### Client-side JavaScript (Website)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Chatbot Updates</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div id="updates"></div>
    
    <script>
        // Connect to Flask-SocketIO server
        const socket = io('http://localhost:5000');
        
        socket.on('connect', () => {
            console.log('Connected to backend');
            socket.emit('subscribe_updates');
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from backend');
        });
        
        // Listen for real-time response updates
        socket.on('response_updated', (data) => {
            console.log('New update:', data);
            
            // Update your UI
            const updatesDiv = document.getElementById('updates');
            const updateElement = document.createElement('div');
            updateElement.innerHTML = `
                <h3>Response ID: ${data.response_id}</h3>
                <p>${data.response}</p>
                <small>Updated: ${data.timestamp}</small>
            `;
            updatesDiv.insertBefore(updateElement, updatesDiv.firstChild);
        });
    </script>
</body>
</html>
```

### React Example

```javascript
import { useEffect, useState } from 'react';
import io from 'socket.io-client';

function ChatbotUpdates() {
    const [updates, setUpdates] = useState([]);
    const [socket, setSocket] = useState(null);
    
    useEffect(() => {
        // Connect to WebSocket
        const newSocket = io('http://localhost:5000');
        
        newSocket.on('connect', () => {
            console.log('Connected');
            newSocket.emit('subscribe_updates');
        });
        
        // Listen for updates
        newSocket.on('response_updated', (data) => {
            setUpdates(prev => [data, ...prev]);
        });
        
        setSocket(newSocket);
        
        // Cleanup
        return () => newSocket.close();
    }, []);
    
    return (
        <div>
            <h2>Real-time Updates</h2>
            {updates.map((update, idx) => (
                <div key={idx}>
                    <h3>{update.response_id}</h3>
                    <p>{update.response}</p>
                    <small>{update.timestamp}</small>
                </div>
            ))}
        </div>
    );
}
```

### Python Client Example

```python
import socketio

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to backend')
    sio.emit('subscribe_updates')

@sio.event
def disconnect():
    print('Disconnected from backend')

@sio.on('response_updated')
def on_response_updated(data):
    print(f"Response updated: {data['response_id']}")
    print(f"Response: {data['response']}")

# Connect to server
sio.connect('http://localhost:5000')

# Keep the connection alive
try:
    sio.wait()
except KeyboardInterrupt:
    sio.disconnect()
```

---

## Complete Examples

### Example 1: Full Workflow

```python
import requests
import time

# 1. Get initial chatbot response
response = requests.post('http://localhost:5000/api/chatbot/query', 
    json={'query': 'I am starting a game. What strategy should I use?'})
initial_data = response.json()
response_id = initial_data['response_id']
print(f"Initial Strategy: {initial_data['response']}")

# 2. Simulate mobile app sending location
location_update = requests.post('http://localhost:5000/api/mobile/update',
    json={
        'response_id': response_id,
        'location': {'latitude': 40.7128, 'longitude': -74.0060},
        'timestamp': '2024-01-01T12:00:00Z'
    })
print("Location updated")

# 3. Simulate pygame sending map update
pygame_update = requests.post('http://localhost:5000/api/pygame/update',
    json={
        'response_id': response_id,
        'update_data': {
            'map_state': 'updated',
            'player_position': {'x': 100, 'y': 200},
            'enemies': [{'id': 1, 'x': 150, 'y': 250}]
        }
    })
updated_data = pygame_update.json()
print(f"Updated Strategy: {updated_data['response']}")

# 4. Get all responses
all_responses = requests.get('http://localhost:5000/api/responses')
print(f"Total responses: {len(all_responses.json()['responses'])}")
```

### Example 2: Integration with Game Loop

```python
import requests
import time

class GameBackend:
    def __init__(self):
        self.api_url = "http://localhost:5000"
        self.response_id = None
        self.strategy = None
    
    def initialize(self):
        """Get initial strategy from chatbot"""
        response = requests.post(f'{self.api_url}/api/chatbot/query',
            json={'query': 'Starting new game session. What strategy?'})
        data = response.json()
        self.response_id = data['response_id']
        self.strategy = data['response']
        return self.strategy
    
    def update_location(self, lat, lon):
        """Update with mobile location"""
        if not self.response_id:
            return
        
        requests.post(f'{self.api_url}/api/mobile/update',
            json={
                'response_id': self.response_id,
                'location': {'latitude': lat, 'longitude': lon},
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
    
    def update_map(self, map_state):
        """Update with pygame map state and get new strategy"""
        payload = {
            'update_data': map_state
        }
        
        if self.response_id:
            payload['response_id'] = self.response_id
        
        response = requests.post(f'{self.api_url}/api/pygame/update',
            json=payload)
        data = response.json()
        
        self.response_id = data['response_id']
        self.strategy = data['response']
        return self.strategy

# Usage
backend = GameBackend()
backend.initialize()

# In your game loop
while game_running:
    map_state = get_current_map_state()
    strategy = backend.update_map(map_state)
    apply_strategy(strategy)
```

---

## Context Management

The chatbot maintains context in two ways:

1. **JSONL File** (`context.jsonl`): Static context loaded on every request
   - Edit this file to provide permanent context to the chatbot
   - Format: Each line is a JSON object with `role` and `content`

2. **Database History**: Previous conversation messages stored in MongoDB
   - The chatbot uses the last 10 messages from previous conversations
   - Context is automatically maintained across requests

**Example context.jsonl:**
```jsonl
{"role": "system", "content": "You are a game strategy advisor."}
{"role": "user", "content": "The game involves tactical combat."}
{"role": "assistant", "content": "I understand. I'll provide tactical combat strategies."}
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing required fields)
- `404`: Not Found (response_id doesn't exist)
- `500`: Internal Server Error

**Example error handling:**
```python
import requests

try:
    response = requests.post('http://localhost:5000/api/chatbot/query',
        json={'query': 'test'}, timeout=10)
    response.raise_for_status()  # Raises exception for bad status codes
    data = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except ValueError:
    print("Invalid JSON response")
```

---

## Testing the Backend

### 1. Test Health Endpoint
```bash
curl http://localhost:5000/api/health
```

### 2. Test Chatbot Query
```bash
curl -X POST http://localhost:5000/api/chatbot/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'
```

### 3. Test WebSocket Connection
Open `http://localhost:5000` in your browser and check the browser console for WebSocket messages.

---

## Tips

1. **Keep response_id**: Store the response_id from the first query to maintain conversation context
2. **Batch updates**: You can batch multiple updates before querying for a new strategy
3. **WebSocket**: Use WebSocket for real-time updates instead of polling
4. **Error handling**: Always handle network errors and timeouts in production
5. **Rate limiting**: Consider implementing rate limiting for production use

---

For more examples, see `API_EXAMPLES.md` in the project directory.

