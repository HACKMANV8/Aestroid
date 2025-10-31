import React from 'react';
import ChatBot from './components/ChatBot';
import NotificationPanel from './components/NotificationPanel';

function App() {
  return (
    <div className="h-screen w-screen bg-black overflow-hidden">
      <main className="h-full w-full p-2 sm:p-4 lg:p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 sm:gap-4 lg:gap-6 h-full max-h-screen">
          {/* Left side - Tactical AI Advisor */}
          <div className="flex flex-col h-full min-h-0">
            <ChatBot />
          </div>

          {/* Right side - Command Center / Alerts */}
          <div className="flex flex-col h-full min-h-0 hidden lg:flex">
            <NotificationPanel />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;