import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ChatMessage } from '../types';

const ChatBot: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  useEffect(() => {
    // Delay scroll to ensure DOM is updated
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [messages, isLoading]);

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
    <div className="flex flex-col h-full bg-[#1a1a1a] rounded-xl border-2 border-orange-500 shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 sm:p-4 border-b-2 border-orange-500/30 flex-shrink-0">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          {/* Signal Icon */}
          <svg className="w-5 h-5 sm:w-6 sm:h-6 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
          </svg>
          <div className="min-w-0">
            <h2 className="text-sm sm:text-base md:text-xl font-bold text-orange-400 uppercase tracking-wide truncate">TACTICAL AI ADVISOR</h2>
            <p className="text-xs text-gray-400 hidden sm:block">Secure Channel • Encrypted</p>
          </div>
        </div>
        {/* Status Dots */}
        <div className="flex gap-1.5 flex-shrink-0">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
        </div>
      </div>
      
      {/* Messages Area - Fixed height with proper overflow */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4 bg-[#1a1a1a] min-h-0">
        {messages.length === 0 ? (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-gray-200 p-3 sm:p-4 rounded-xl max-w-[85%] sm:max-w-md shadow-lg break-words">
              <p className="mb-2 break-words">Tactical AI online. Command Center ready. How may I assist with your strategic planning?</p>
              <p className="text-xs text-gray-400">{formatTime(new Date())}</p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className="space-y-3">
              {/* User Message */}
              <div className="flex justify-end">
                <div className="bg-gray-700 text-gray-200 p-3 rounded-xl max-w-[85%] sm:max-w-md shadow-lg break-words">
                  <p className="break-words">{message.userMessage}</p>
                  <p className="text-xs text-gray-400 mt-1">{formatTime(message.timestamp)}</p>
                </div>
              </div>
              {/* AI Response */}
              <div className="flex justify-start">
                <div className="bg-gray-700 text-gray-200 p-3 rounded-xl max-w-[85%] sm:max-w-md shadow-lg break-words">
                  <p className="whitespace-pre-wrap break-words">{message.botResponse}</p>
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
        <div ref={messagesEndRef} className="h-1" />
      </div>

      {/* Input Area */}
      <form onSubmit={sendMessage} className="p-3 sm:p-4 border-t-2 border-orange-500/30 bg-[#1a1a1a] flex-shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Enter tactical command..."
            className="flex-1 bg-gray-800 text-gray-200 placeholder-gray-500 p-2 sm:p-3 rounded-lg border-2 border-orange-500/30 focus:outline-none focus:border-orange-500 focus:ring-2 focus:ring-orange-500/20 text-sm sm:text-base"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="bg-orange-500 hover:bg-orange-600 text-white p-2 sm:p-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg hover:shadow-orange-500/50 flex items-center justify-center min-w-[40px] sm:min-w-[48px] flex-shrink-0"
            title="Send message"
          >
            <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBot;