# Military Location Tracking System

A real-time military location tracking system with AI chat integration using Node.js, TypeScript, React, and MongoDB.

## Features

- **Real-time Location Tracking**: Receives location data from Android apps via REST API
- **Live WebSocket Alerts**: Broadcasts location updates to all connected clients
- **AI Chat Integration**: Groq API-powered chatbot for military assistance
- **MongoDB Storage**: Persistent storage for chat history and location data
- **Responsive UI**: Clean dashboard with chat interface and notification panel

## Tech Stack

### Backend
- Node.js + TypeScript + Express
- MongoDB with Mongoose
- Socket.IO for WebSocket communication
- Groq SDK for AI chat
- CORS enabled for cross-origin requests

### Frontend
- React + TypeScript
- Tailwind CSS for styling
- Socket.IO client for real-time updates
- Axios for HTTP requests

## Quick Start

### Prerequisites
- Node.js (v16 or higher)
- MongoDB running on localhost:27017
- npm or yarn

### Installation

1. **Install backend dependencies:**
   ```bash
   npm install
   ```

2. **Install frontend dependencies:**
   ```bash
   cd client
   npm install
   cd ..
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your actual values:
   ```
   PORT=5000
   MONGO_URI=mongodb://localhost:27017/military-tracker
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Application

1. **Start both backend and frontend:**
   ```bash
   npm run dev
   ```

   Or run them separately:
   ```bash
   # Terminal 1 - Backend
   npm run server:dev
   
   # Terminal 2 - Frontend
   npm run client:dev
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## API Endpoints

### Chat Endpoints
- `POST /api/chat` - Send message to AI chatbot
- `GET /api/chat/history` - Get chat history

### Location Endpoints
- `POST /api/location` - Receive location data from Android app
- `GET /api/locations` - Get all stored locations

### Health Check
- `GET /health` - Server health status

## Testing Location Updates

Use the included test script to simulate Android app location posts:

```bash
node test-location.js
```

This will send mock location data to the API and trigger WebSocket alerts.

## Android App Integration

Your Android app should POST location data to `/api/location` with this format:

```json
{
  "unitId": "TANK-001",
  "unitType": "tank",
  "latitude": 12.9716,
  "longitude": 77.5946
}
```

## Database Schema

### Chats Collection
```javascript
{
  userMessage: String,
  botResponse: String,
  timestamp: Date
}
```

### Locations Collection
```javascript
{
  unitId: String,
  unitType: String,
  latitude: Number,
  longitude: Number,
  timestamp: Date
}
```

## WebSocket Events

- `connection` - Client connects
- `disconnect` - Client disconnects  
- `locationAlert` - New location received (broadcasted to all clients)

## Project Structure

```
├── src/
│   ├── controllers/
│   │   ├── chatController.ts
│   │   └── locationController.ts
│   ├── models/
│   │   ├── Chat.ts
│   │   └── Location.ts
│   ├── routes/
│   │   ├── chatRoutes.ts
│   │   └── locationRoutes.ts
│   └── server.ts
├── client/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBot.tsx
│   │   │   └── NotificationPanel.tsx
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   └── package.json
├── test-location.js
├── package.json
└── README.md
```

## Development

- Backend runs on port 5000
- Frontend runs on port 3000
- MongoDB connection: localhost:27017
- WebSocket server ready for real-time communication

## Production Deployment

1. Build the frontend:
   ```bash
   npm run client:build
   ```

2. Build the backend:
   ```bash
   npm run build
   ```

3. Start production server:
   ```bash
   npm start
   ```

## Troubleshooting

- Ensure MongoDB is running before starting the server
- Check that all environment variables are set correctly
- Verify Groq API key is valid
- Make sure ports 3000 and 5000 are available