import { Request, Response } from 'express';
import Groq from 'groq-sdk';
import Chat from '../models/Chat';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Validate API key on startup
const GROQ_API_KEY = process.env.GROQ_API_KEY;
let groq: Groq | null = null;

if (!GROQ_API_KEY) {
  console.warn('⚠️  WARNING: GROQ_API_KEY is not set in environment variables');
  console.warn('⚠️  Chat functionality will not work without a valid API key');
} else {
  try {
    groq = new Groq({
      apiKey: GROQ_API_KEY,
    });
    console.log('✅ Groq client initialized');
  } catch (error) {
    console.error('❌ Failed to initialize Groq client:', error);
  }
}

export const sendMessage = async (req: Request, res: Response) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Check if API key is configured
    if (!GROQ_API_KEY || !groq) {
      return res.status(500).json({ 
        error: 'Groq API key is not configured. Please set GROQ_API_KEY in your .env file.' 
      });
    }

    // Send message to Groq API
    const completion = await groq.chat.completions.create({
      messages: [
        {
          role: 'user',
          content: message,
        },
      ],
      model: 'llama-3.3-70b-versatile',
    });

    const botResponse = completion.choices[0]?.message?.content || 'No response from AI';

    // Save chat to database (only if MongoDB is connected)
    try {
      const chat = new Chat({
        userMessage: message,
        botResponse,
      });
      await chat.save();
    } catch (dbError) {
      // Log but don't fail if database save fails
      console.warn('Failed to save chat to database (MongoDB may not be connected):', dbError);
    }

    res.json({
      userMessage: message,
      botResponse,
      timestamp: new Date(),
    });
  } catch (error: any) {
    console.error('Chat error:', error);
    
    // Provide more specific error messages
    if (error?.status === 401 || error?.message?.includes('authentication')) {
      return res.status(401).json({ 
        error: 'Invalid Groq API key. Please check your GROQ_API_KEY in .env file.' 
      });
    }
    
    if (error?.status === 429) {
      return res.status(429).json({ 
        error: 'Rate limit exceeded. Please try again later.' 
      });
    }

    res.status(500).json({ 
      error: 'Failed to process chat message',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

export const getChatHistory = async (req: Request, res: Response) => {
  try {
    const chats = await Chat.find()
      .sort({ timestamp: -1 })
      .limit(50);

    res.json(chats.reverse());
  } catch (error) {
    console.error('Get chat history error:', error);
    res.status(500).json({ error: 'Failed to fetch chat history' });
  }
};