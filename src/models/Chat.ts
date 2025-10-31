import mongoose, { Document, Schema } from 'mongoose';

export interface IChat extends Document {
  userMessage: string;
  botResponse: string;
  timestamp: Date;
}

const ChatSchema: Schema = new Schema({
  userMessage: {
    type: String,
    required: true,
  },
  botResponse: {
    type: String,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

export default mongoose.model<IChat>('Chat', ChatSchema);