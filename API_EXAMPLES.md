# API Usage Examples

## 1. Chatbot Query

Get a response from the chatbot:

```bash
curl -X POST http://localhost:5000/api/chatbot/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the strategy for this situation?"
  }'
```

Response:
```json
{
  "response_id": "uuid-here",
  "response": "Chatbot response text...",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

## 2. Mobile App Update

Update a response with location and timestamp from mobile app:

```bash
curl -X POST http://localhost:5000/api/mobile/update \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "uuid-here",
    "location": {
      "latitude": 40.7128,
      "longitude": -74.0060
    },
    "timestamp": "2024-01-01T12:00:00Z"
  }'
```

## 3. Pygame Update

Send updates from pygame application:

```bash
curl -X POST http://localhost:5000/api/pygame/update \
  -H "Content-Type: application/json" \
  -d '{
    "update_data": {
      "map_state": "updated",
      "player_position": {"x": 100, "y": 200},
      "events": ["enemy_spawned", "item_collected"]
    },
    "response_id": "uuid-here"
  }'
```

Or create a new response:

```bash
curl -X POST http://localhost:5000/api/pygame/update \
  -H "Content-Type: application/json" \
  -d '{
    "update_data": {
      "map_state": "initial",
      "player_position": {"x": 0, "y": 0}
    }
  }'
```

## 4. Get All Responses

```bash
curl http://localhost:5000/api/responses
```

## 5. Get Specific Response

```bash
curl http://localhost:5000/api/responses/{response_id}
```

## 6. Health Check

```bash
curl http://localhost:5000/api/health
```

## Python Example

```python
import requests
import json

# Chatbot query
response = requests.post('http://localhost:5000/api/chatbot/query', 
    json={'query': 'What should I do next?'})
data = response.json()
response_id = data['response_id']

# Mobile app update
requests.post('http://localhost:5000/api/mobile/update',
    json={
        'response_id': response_id,
        'location': {'latitude': 40.7128, 'longitude': -74.0060},
        'timestamp': '2024-01-01T12:00:00Z'
    })

# Pygame update
requests.post('http://localhost:5000/api/pygame/update',
    json={
        'response_id': response_id,
        'update_data': {'map_state': 'updated', 'events': ['update']}
    })
```

## JavaScript Example (for website)

```javascript
// Using fetch API
async function queryChatbot(query) {
    const response = await fetch('http://localhost:5000/api/chatbot/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    });
    const data = await response.json();
    return data;
}

// Update with mobile location
async function updateMobile(responseId, location, timestamp) {
    const response = await fetch('http://localhost:5000/api/mobile/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            response_id: responseId,
            location: location,
            timestamp: timestamp
        })
    });
    const data = await response.json();
    return data;
}

// Pygame update
async function updatePygame(updateData, responseId = null) {
    const response = await fetch('http://localhost:5000/api/pygame/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            update_data: updateData,
            response_id: responseId
        })
    });
    const data = await response.json();
    return data;
}
```

