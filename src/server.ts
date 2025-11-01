import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import { createServer } from 'http';
import { Server } from 'socket.io';
import fs from 'fs';
import path from 'path';
import multer from 'multer';
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

// Path for storing tactical data
const TACTICAL_DATA_PATH = path.join(process.cwd(), 'data', 'tactical_data.json');

// Ensure data directory exists
const dataDir = path.join(process.cwd(), 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// Ensure images directory exists
const imagesDir = path.join(process.cwd(), 'data', 'images');
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
}

// Configure multer for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, imagesDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'tactical-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Only PNG images are allowed'));
    }
  }
});

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

// Tactical data endpoints
app.post('/api/tactical-data', (req, res) => {
  try {
    const tacticalData = req.body;
    
    // Validate required fields
    if (!tacticalData.timestamp || !tacticalData.markers) {
      return res.status(400).json({ 
        error: 'Invalid data format. Required fields: timestamp, markers' 
      });
    }

    // Add server timestamp
    tacticalData.received_at = new Date().toISOString();

    // Write to file
    fs.writeFileSync(TACTICAL_DATA_PATH, JSON.stringify(tacticalData, null, 2));

    console.log(`✅ Tactical data saved: ${tacticalData.marker_count} markers at ${new Date(tacticalData.timestamp * 1000).toISOString()}`);

    // Emit update via Socket.IO
    io.emit('tactical-data-update', tacticalData);

    res.json({ 
      success: true, 
      message: 'Tactical data stored successfully',
      marker_count: tacticalData.marker_count,
      timestamp: tacticalData.timestamp
    });
  } catch (error: any) {
    console.error('Error storing tactical data:', error);
    res.status(500).json({ 
      error: 'Failed to store tactical data',
      details: error.message 
    });
  }
});

app.get('/api/tactical-data', (req, res) => {
  try {
    // Check if file exists
    if (!fs.existsSync(TACTICAL_DATA_PATH)) {
      return res.status(404).json({ 
        error: 'No tactical data available',
        message: 'No data has been stored yet'
      });
    }

    // Read and parse the file
    const data = fs.readFileSync(TACTICAL_DATA_PATH, 'utf-8');
    const tacticalData = JSON.parse(data);

    res.json(tacticalData);
  } catch (error: any) {
    console.error('Error reading tactical data:', error);
    res.status(500).json({ 
      error: 'Failed to read tactical data',
      details: error.message 
    });
  }
});

// Image upload endpoint
app.post('/api/upload-image', upload.single('image'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No image file provided' 
      });
    }

    console.log(`✅ Image uploaded: ${req.file.filename} (${(req.file.size / 1024).toFixed(2)} KB)`);

    // Store metadata
    const imageMetadata = {
      filename: req.file.filename,
      originalName: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
      uploadedAt: new Date().toISOString(),
      path: `/api/images/${req.file.filename}`
    };

    // Emit via Socket.IO
    io.emit('image-uploaded', imageMetadata);

    res.json({
      success: true,
      message: 'Image uploaded successfully',
      image: imageMetadata
    });
  } catch (error: any) {
    console.error('Error uploading image:', error);
    res.status(500).json({ 
      error: 'Failed to upload image',
      details: error.message 
    });
  }
});

// Get specific image
app.get('/api/images/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const imagePath = path.join(imagesDir, filename);

    // Check if file exists
    if (!fs.existsSync(imagePath)) {
      return res.status(404).json({ 
        error: 'Image not found' 
      });
    }

    // Verify it's a PNG
    if (!filename.endsWith('.png')) {
      return res.status(400).json({ 
        error: 'Invalid image format' 
      });
    }

    // Send the image file
    res.sendFile(imagePath);
  } catch (error: any) {
    console.error('Error retrieving image:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve image',
      details: error.message 
    });
  }
});

// List all uploaded images
app.get('/api/images', (req, res) => {
  try {
    const files = fs.readdirSync(imagesDir);
    const pngFiles = files.filter(f => f.endsWith('.png'));
    
    const images = pngFiles.map(filename => {
      const filePath = path.join(imagesDir, filename);
      const stats = fs.statSync(filePath);
      
      return {
        filename,
        size: stats.size,
        uploadedAt: stats.mtime,
        path: `/api/images/${filename}`
      };
    });

    res.json({
      count: images.length,
      images
    });
  } catch (error: any) {
    console.error('Error listing images:', error);
    res.status(500).json({ 
      error: 'Failed to list images',
      details: error.message 
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'Server is running', 
    timestamp: new Date().toISOString(),
    tactical_data_available: fs.existsSync(TACTICAL_DATA_PATH)
  });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send current tactical data to newly connected client
  if (fs.existsSync(TACTICAL_DATA_PATH)) {
    try {
      const data = fs.readFileSync(TACTICAL_DATA_PATH, 'utf-8');
      const tacticalData = JSON.parse(data);
      socket.emit('tactical-data-update', tacticalData);
    } catch (error: any) {
      console.error('Error sending tactical data to new client:', error);
    }
  }
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Start server regardless of MongoDB connection
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`WebSocket server ready for connections`);
  console.log(`Tactical data storage: ${TACTICAL_DATA_PATH}`);
});

// Connect to MongoDB (non-blocking)
mongoose.connect(MONGO_URI)
  .then(() => {
    console.log('Connected to MongoDB');
  })
  .catch((error: any) => {
    console.error('MongoDB connection error:', error.message);
    console.warn('⚠️  Server running without MongoDB. Database features will not work.');
    console.warn('⚠️  Please ensure MongoDB is running on', MONGO_URI);
  });

