import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ChatMessage } from '../types';

const ChatBot: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [strategyAnalysis, setStrategyAnalysis] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);
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
    loadLatestImage();
  }, []);

  const loadChatHistory = async () => {
    try {
      const response = await axios.get('/api/chat/history');
      setMessages(response.data);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const loadLatestImage = async () => {
    try {
      const response = await axios.get('/api/images');
      if (response.data.images && response.data.images.length > 0) {
        // Sort by upload time and get the latest
        const sortedImages = response.data.images.sort((a: any, b: any) => 
          new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
        );
        setLatestImage(sortedImages[0].path);
        setImageError(false);
      }
    } catch (error) {
      console.error('Failed to load latest image:', error);
      setImageError(true);
    }
  };

  const analyzeTacticalData = async () => {
    setIsAnalyzing(true);
    setShowStrategyModal(true);
    setStrategyAnalysis('');

    try {
      // Fetch current tactical data
      const tacticalResponse = await axios.get('/api/tactical-data');
      const tacticalData = tacticalResponse.data;

      // Create strategy request prompt
      const strategyPrompt = `Analyze the following battlefield tactical data and provide a comprehensive military strategy:

Battlefield Status:
- Total Markers: ${tacticalData.marker_count}
- Timestamp: ${new Date(tacticalData.timestamp * 1000).toLocaleString()}
- Camera Orientation: Yaw ${tacticalData.yaw_deg}°, Roll ${tacticalData.roll_deg}°

Unit Composition:
${tacticalData.markers.map((m: any) => 
  `- ${m.id}: ${m.team.toUpperCase()} ${m.type === '1' ? 'Infantry' : 'Vehicle'} at (${m.lat.toFixed(4)}, ${m.lon.toFixed(4)}, ${m.elev}m)`
).join('\n')}

Provide:
1. Situation Assessment
2. Threat Analysis
3. Recommended Strategy
4. Priority Actions
5. Risk Factors`;

      // Send to AI for analysis
      const aiResponse = await axios.post('/api/chat', { 
        message: strategyPrompt 
      });

      setStrategyAnalysis(aiResponse.data.botResponse);
    } catch (error: any) {
      console.error('Failed to analyze tactical data:', error);
      if (error.response?.status === 404) {
        setStrategyAnalysis('⚠️ NO TACTICAL DATA AVAILABLE\n\nNo battlefield data has been received yet. Awaiting reconnaissance input.');
      } else {
        setStrategyAnalysis('❌ TACTICAL ANALYSIS FAILED\n\nUnable to process battlefield data. System offline or communication error.');
      }
    } finally {
      setIsAnalyzing(false);
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
        <div className="flex gap-2 items-center">
          <button
            onClick={analyzeTacticalData}
            disabled={isAnalyzing}
            className="bg-green-600 hover:bg-green-500 text-white font-bold px-3 py-1.5 rounded-lg shadow-[0_0_15px_rgba(34,197,94,0.4)] transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed text-xs md:text-sm uppercase tracking-wide"
            title="Analyze current battlefield tactical data"
          >
            {isAnalyzing ? '⟳ ANALYZING...' : '⚡ TACTICAL ANALYSIS'}
          </button>
          <div className="flex gap-2">
            <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse" />
            <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse delay-100" />
            <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse delay-200" />
          </div>
        </div>
      </div>

      {/* Latest Tactical Image */}
      {latestImage && !imageError && (
        <div className="border-b border-orange-500/40 bg-black/50 p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-bold text-orange-400 uppercase tracking-wide flex items-center gap-2">
              <span>📡</span> LATEST TACTICAL RECONNAISSANCE
            </h3>
            <button
              onClick={loadLatestImage}
              className="text-orange-400 hover:text-orange-300 text-xs uppercase tracking-wide"
              title="Refresh image"
            >
              🔄 Refresh
            </button>
          </div>
          <div className="bg-[#1a0d00] border border-orange-500/30 rounded-lg overflow-hidden shadow-[0_0_15px_rgba(255,140,0,0.2)]">
            <img 
              src={`http://localhost:5000${latestImage}`}
              alt="Tactical reconnaissance"
              className="w-full h-auto max-h-64 object-contain"
              onError={() => setImageError(true)}
            />
          </div>
        </div>
      )}

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

      {/* Strategy Analysis Modal */}
      {showStrategyModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-[#0d0d0d] border-2 border-green-500/60 rounded-2xl shadow-[0_0_40px_rgba(34,197,94,0.5)] max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className="bg-gradient-to-r from-green-900/50 to-green-800/50 p-4 border-b border-green-500/50 flex justify-between items-center">
              <h3 className="text-xl font-bold text-green-400 tracking-widest uppercase flex items-center gap-2">
                <span className="text-2xl">🎯</span> TACTICAL STRATEGY ANALYSIS
              </h3>
              <button
                onClick={() => setShowStrategyModal(false)}
                className="text-green-400 hover:text-green-300 text-2xl font-bold leading-none"
                title="Close"
              >
                ×
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-black via-[#001a00] to-black">
              <div className="relative">
                <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle,rgba(34,197,94,0.3)_1px,transparent_1px)] bg-[length:20px_20px]" />
                
                <div className="relative z-10">
                  {isAnalyzing ? (
                    <div className="flex flex-col items-center justify-center py-12 space-y-4">
                      <div className="flex space-x-2">
                        <div className="w-4 h-4 bg-green-500 rounded-full animate-bounce"></div>
                        <div className="w-4 h-4 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-4 h-4 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <p className="text-green-400 font-mono text-lg">ANALYZING BATTLEFIELD DATA...</p>
                      <p className="text-green-600 text-sm">Processing tactical information and generating strategy</p>
                    </div>
                  ) : (
                    <div className="bg-[#001a00] border border-green-500/30 rounded-xl p-6 shadow-[0_0_20px_rgba(34,197,94,0.2)]">
                      <pre className="text-green-300 font-mono text-sm leading-relaxed whitespace-pre-wrap">
                        {strategyAnalysis || 'No analysis available'}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="bg-black/70 p-4 border-t border-green-500/50 flex justify-end">
              <button
                onClick={() => setShowStrategyModal(false)}
                className="bg-green-600 hover:bg-green-500 text-white font-bold px-6 py-2 rounded-lg shadow-[0_0_15px_rgba(34,197,94,0.4)] transition-all active:scale-95 uppercase tracking-wide"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatBot;

