import mongoose, { Document, Schema } from 'mongoose';

export interface ILocation extends Document {
  unitId: string;
  unitType: string;
  latitude: number;
  longitude: number;
  timestamp: Date;
}

const LocationSchema: Schema = new Schema({
  unitId: {
    type: String,
    required: true,
  },
  unitType: {
    type: String,
    required: true,
  },
  latitude: {
    type: Number,
    required: true,
  },
  longitude: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

export default mongoose.model<ILocation>('Location', LocationSchema);