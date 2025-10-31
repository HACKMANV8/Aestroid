import React, { useState, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';
import { LocationAlert } from '../types';

const NotificationPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<LocationAlert[]>([]);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
      setIsConnected(false);
    });

    newSocket.on('locationAlert', (alert: LocationAlert) => {
      console.log('Received location alert:', alert);
      setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep only last 10 alerts
    });

    return () => {
      newSocket.close();
    };
  }, []);

  const clearAlerts = () => {
    setAlerts([]);
  };

  const removeAlert = (index: number) => {
    setAlerts(prev => prev.filter((_, i) => i !== index));
  };

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleString();
  };

  return (
    <div className="flex flex-col h-full bg-[#1a1a1a] rounded-xl border-2 border-orange-500 shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 sm:p-4 border-b-2 border-orange-500/30 flex-shrink-0">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <svg className="w-5 h-5 sm:w-6 sm:h-6 text-orange-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
          <div className="min-w-0">
            <h2 className="text-sm sm:text-base md:text-xl font-bold text-orange-400 uppercase tracking-wide truncate">Command Center</h2>
            <p className="text-xs text-gray-400 hidden sm:block">Live Intelligence Feed</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
          <span className="text-xs text-gray-400 hidden sm:inline">{isConnected ? 'Active' : 'Offline'}</span>
        </div>
      </div>

      {/* Alerts Area - Fixed height with proper overflow */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 bg-[#1a1a1a] min-h-0">
        {alerts.length === 0 ? (
          <div className="text-gray-400 text-center py-12">
            <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            <div className="text-lg font-semibold mb-2">No Active Alerts</div>
            <div className="text-sm">
              Awaiting tactical intelligence...
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {alerts.map((alert, index) => (
              <div
                key={index}
                className="bg-gray-800 border-l-4 border-orange-500 p-4 rounded-r-lg relative group shadow-lg hover:bg-gray-750 transition-colors"
              >
                <button
                  onClick={() => removeAlert(index)}
                  className="absolute top-2 right-2 text-gray-400 hover:text-orange-400 opacity-0 group-hover:opacity-100 transition-opacity text-xl leading-none"
                  title="Dismiss alert"
                >
                  ×
                </button>
                <div className="text-orange-400 font-medium mb-2 flex items-center gap-2">
                  <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse"></div>
                  {alert.message}
                </div>
                <div className="text-gray-300 text-sm space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Unit:</span>
                    <span className="font-mono">{alert.unitId}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Type:</span>
                    <span className="uppercase">{alert.unitType.replace('_', ' ')}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Coordinates:</span>
                    <span className="font-mono text-orange-400">{typeof alert.latitude === 'number' ? alert.latitude.toFixed(4) : alert.latitude}, {typeof alert.longitude === 'number' ? alert.longitude.toFixed(4) : alert.longitude}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-700">
                    {formatTime(alert.timestamp)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {alerts.length > 0 && (
        <div className="p-3 sm:p-4 border-t-2 border-orange-500/30 bg-[#1a1a1a] flex-shrink-0">
          <button
            onClick={clearAlerts}
            className="w-full bg-orange-500 hover:bg-orange-600 text-white py-2 rounded-lg transition-colors font-semibold uppercase tracking-wide shadow-lg hover:shadow-orange-500/50 text-sm sm:text-base"
          >
            Clear All ({alerts.length})
          </button>
        </div>
      )}
    </div>
  );
};

export default NotificationPanel;