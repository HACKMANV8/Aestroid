import express from 'express';
import { receiveLocation, getLocations } from '../controllers/locationController';

const router = express.Router();

router.post('/location', receiveLocation);
router.get('/locations', getLocations);

export default router;