import React from 'react';
import ChatBot from './components/ChatBot';
import NotificationPanel from './components/NotificationPanel';

function App() {
  return (
    <div className="min-h-screen bg-black">
      <main className="container mx-auto p-4 lg:p-6 h-screen">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6 h-full">
          {/* Left side - Tactical AI Advisor */}
          <div className="flex flex-col h-full">
            <ChatBot />
          </div>

          {/* Right side - Command Center / Alerts */}
          <div className="flex flex-col h-full">
            <NotificationPanel />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;