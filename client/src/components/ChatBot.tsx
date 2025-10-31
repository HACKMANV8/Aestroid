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
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [messages, isLoading]);

  useEffect(() => {
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
      const response = await axios.post('/api/chat', { message: userMessage });

      const newMessage: ChatMessage = {
        userMessage: response.data.userMessage,
        botResponse: response.data.botResponse,
        timestamp: response.data.timestamp ? new Date(response.data.timestamp) : new Date(),
      };

      setMessages((prev) => [...prev, newMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage: ChatMessage = {
        userMessage: userMessage,
        botResponse: 'Error: Tactical analyzer offline. Please retry command.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
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
    <div className="flex flex-col h-full bg-[#0d0d0d] rounded-2xl border border-orange-500/40 shadow-[0_0_25px_rgba(255,140,0,0.3)] overflow-hidden font-mono">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-orange-500/50 bg-black/70">
        <h2 className="text-lg md:text-xl font-bold text-orange-400 tracking-widest uppercase">
          WAR STRATEGY ANALYZER
        </h2>
        <div className="flex gap-2">
          <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse" />
          <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse delay-100" />
          <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse delay-200" />
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 relative bg-gradient-to-b from-black via-[#1a0d00] to-black">
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle,rgba(255,140,0,0.25)_1px,transparent_1px)] bg-[length:20px_20px]" />
        <div className="relative z-10 space-y-4">
          {messages.length === 0 ? (
            <div className="flex justify-start">
              <div className="bg-[#1a0d00] text-orange-300 p-4 rounded-xl max-w-[85%] shadow-[0_0_15px_rgba(255,140,0,0.4)] border border-orange-400/40">
                <p className="mb-2">
                  Tactical AI online. Command Center operational. Awaiting strategic directive.
                </p>
                <p className="text-xs text-orange-500/60">{formatTime(new Date())}</p>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className="space-y-3">
                {/* Commander (User) Message */}
                <div className="flex justify-end">
                  <div className="bg-[#2c1a00] text-gray-100 p-3 sm:p-4 rounded-xl max-w-[80%] shadow-[0_0_12px_rgba(255,165,0,0.3)] border border-orange-400/40">
                    <p className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">
                      {message.userMessage}
                    </p>
                    <p className="text-xs text-orange-500/50 mt-1 text-right">
                      {formatTime(message.timestamp)}
                    </p>
                  </div>
                </div>

                {/* Tactical AI Response */}
                <div className="flex justify-start">
                  <div className="bg-[#1a0d00] text-orange-300 p-3 sm:p-4 rounded-xl max-w-[85%] shadow-[0_0_18px_rgba(255,140,0,0.3)] border border-orange-400/40 font-mono leading-relaxed">
                    {message.botResponse.split('\n').map((line, i) => (
                      <span key={i}>
                        {line.startsWith('>') ? (
                          <span className="text-orange-400 font-semibold">{line}</span>
                        ) : (
                          line
                        )}
                        <br />
                      </span>
                    ))}
                    <p className="text-xs text-orange-500/60 mt-1">
                      {formatTime(message.timestamp)}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-[#1a0d00] text-orange-300 p-3 rounded-xl border border-orange-400/40 shadow-[0_0_10px_rgba(255,140,0,0.3)]">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.1s' }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.2s' }}
                  ></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-1" />
        </div>
      </div>

      {/* Input Area */}
      <form
        onSubmit={sendMessage}
        className="p-4 border-t border-orange-500/40 bg-black/80 flex-shrink-0"
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Enter tactical command..."
            className="flex-1 bg-[#1a0d00] text-orange-400 placeholder-orange-700 p-3 rounded-lg border border-orange-500/30 focus:outline-none focus:ring-2 focus:ring-orange-400/40 font-mono text-sm md:text-base"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="bg-orange-500 hover:bg-orange-400 text-black font-bold px-4 py-2 rounded-lg shadow-[0_0_10px_rgba(255,140,0,0.3)] transition-transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Send message"
          >
            SEND
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBot;
