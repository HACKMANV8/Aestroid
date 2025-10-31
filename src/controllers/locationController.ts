import { Request, Response } from 'express';
import Location from '../models/Location';
import { Server } from 'socket.io';

let io: Server;

export const setSocketIO = (socketIO: Server) => {
  io = socketIO;
};

export const receiveLocation = async (req: Request, res: Response) => {
  try {
    const { unitId, unitType, latitude, longitude } = req.body;

    if (!unitId || !unitType || latitude === undefined || longitude === undefined) {
      return res.status(400).json({ 
        error: 'unitId, unitType, latitude, and longitude are required' 
      });
    }

    // Save location to database
    const location = new Location({
      unitId,
      unitType,
      latitude,
      longitude,
    });

    await location.save();

    // Broadcast to all connected clients via WebSocket
    if (io) {
      const alertMessage = `🪖 New ${unitType} position at (${latitude}, ${longitude})`;
      io.emit('locationAlert', {
        message: alertMessage,
        unitId,
        unitType,
        latitude,
        longitude,
        timestamp: location.timestamp,
      });
    }

    res.json({
      message: 'Location received successfully',
      location: {
        unitId,
        unitType,
        latitude,
        longitude,
        timestamp: location.timestamp,
      },
    });
  } catch (error) {
    console.error('Location error:', error);
    res.status(500).json({ error: 'Failed to process location data' });
  }
};

export const getLocations = async (req: Request, res: Response) => {
  try {
    const locations = await Location.find()
      .sort({ timestamp: -1 })
      .limit(100);

    res.json(locations);
  } catch (error) {
    console.error('Get locations error:', error);
    res.status(500).json({ error: 'Failed to fetch locations' });
  }
};