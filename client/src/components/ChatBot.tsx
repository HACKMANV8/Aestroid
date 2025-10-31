import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ChatMessage } from '../types';

const ChatBot: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load chat history on component mount
    loadChatHistory();
  }, []);

  const loadChatHistory = async () => {
    try {
      const response = await axios.get('/api/chat/history');
      setMessages(response.data);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    setIsLoading(true);
    const userMessage = inputMessage;
    setInputMessage('');

    try {
      const response = await axios.post('/api/chat', {
        message: userMessage,
      });

      const newMessage: ChatMessage = {
        userMessage: response.data.userMessage,
        botResponse: response.data.botResponse,
        timestamp: response.data.timestamp ? new Date(response.data.timestamp) : new Date(),
      };

      setMessages(prev => [...prev, newMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Add error message to chat
      const errorMessage: ChatMessage = {
        userMessage: userMessage,
        botResponse: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date: Date | string) => {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    const hours = dateObj.getHours();
    const minutes = dateObj.getMinutes();
    const ampm = hours >= 12 ? 'pm' : 'am';
    const formattedHours = hours % 12 || 12;
    const formattedMinutes = minutes.toString().padStart(2, '0');
    return `${formattedHours}:${formattedMinutes} ${ampm}`;
  };

  return (
    <div className="flex flex-col h-full bg-[#1a1a1a] rounded-xl border-2 border-orange-500 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b-2 border-orange-500/30">
        <div className="flex items-center gap-3">
          {/* Signal Icon */}
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
          </svg>
          <div>
            <h2 className="text-xl font-bold text-orange-400 uppercase tracking-wide">TACTICAL AI ADVISOR</h2>
            <p className="text-xs text-gray-400">Secure Channel • Encrypted</p>
          </div>
        </div>
        {/* Status Dots */}
        <div className="flex gap-1.5">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
        </div>
      </div>
      
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#1a1a1a]">
        {messages.length === 0 ? (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-gray-200 p-4 rounded-xl max-w-md shadow-lg">
              <p className="mb-2">Tactical AI online. Command Center ready. How may I assist with your strategic planning?</p>
              <p className="text-xs text-gray-400">{formatTime(new Date())}</p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className="space-y-3">
              {/* User Message */}
              <div className="flex justify-end">
                <div className="bg-gray-700 text-gray-200 p-3 rounded-xl max-w-md shadow-lg">
                  <p>{message.userMessage}</p>
                  <p className="text-xs text-gray-400 mt-1">{formatTime(message.timestamp)}</p>
                </div>
              </div>
              {/* AI Response */}
              <div className="flex justify-start">
                <div className="bg-gray-700 text-gray-200 p-3 rounded-xl max-w-md shadow-lg">
                  <p className="whitespace-pre-wrap">{message.botResponse}</p>
                  <p className="text-xs text-gray-400 mt-1">{formatTime(message.timestamp)}</p>
                </div>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-gray-200 p-3 rounded-xl">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={sendMessage} className="p-4 border-t-2 border-orange-500/30 bg-[#1a1a1a]">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Enter tactical command..."
            className="flex-1 bg-gray-800 text-gray-200 placeholder-gray-500 p-3 rounded-lg border-2 border-orange-500/30 focus:outline-none focus:border-orange-500 focus:ring-2 focus:ring-orange-500/20"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="bg-orange-500 hover:bg-orange-600 text-white p-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg hover:shadow-orange-500/50 flex items-center justify-center min-w-[48px]"
            title="Send message"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBot;