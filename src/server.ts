import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import { createServer } from 'http';
import { Server } from 'socket.io';
import chatRoutes from './routes/chatRoutes';
import locationRoutes from './routes/locationRoutes';
import { setSocketIO } from './controllers/locationController';

dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: (origin, callback) => {
      // Allow all origins
      callback(null, true);
    },
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: false,
    allowedHeaders: ["Content-Type", "Authorization"]
  }
});

const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/war';
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Validate environment variables on startup
if (!GROQ_API_KEY) {
  console.warn('⚠️  WARNING: GROQ_API_KEY is not set in environment variables');
  console.warn('⚠️  Chat functionality will not work. Please add GROQ_API_KEY to your .env file');
} else {
  console.log('✅ Groq API key configured');
}

// Middleware - CORS configured to allow all origins
app.use(cors({
  origin: '*', // Allow all origins
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: false
}));

// Handle preflight requests
app.options('*', cors());

app.use(express.json());

// Set up Socket.IO for location controller
setSocketIO(io);

// Routes
app.use('/api', chatRoutes);
app.use('/api', locationRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'Server is running', timestamp: new Date().toISOString() });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Start server regardless of MongoDB connection
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`WebSocket server ready for connections`);
});

// Connect to MongoDB (non-blocking)
mongoose.connect(MONGO_URI)
  .then(() => {
    console.log('Connected to MongoDB');
  })
  .catch((error) => {
    console.error('MongoDB connection error:', error.message);
    console.warn('⚠️  Server running without MongoDB. Database features will not work.');
    console.warn('⚠️  Please ensure MongoDB is running on', MONGO_URI);
  });